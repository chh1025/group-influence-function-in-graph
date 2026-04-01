import logging
import math
from typing import Callable, Optional

import numpy as np
import scipy.sparse.linalg as L
import torch
from torch import nn
from torch.utils import data

from torch_influence.base import BaseInfluenceModule, BaseObjective
import torch_geometric
import sys

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()

class AutogradInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    by directly forming and inverting the risk Hessian matrix using :mod:`torch.autograd`
    utilities.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        check_eigvals: if ``True``, this initializer checks that the damped risk Hessian
            is positive definite, and raises a :mod:`ValueError` if it is not. Otherwise,
            no check is performed.

    Warnings:
        This module scales poorly with the number of model parameters :math:`d`. In
        general, computing the Hessian matrix takes :math:`\mathcal{O}(nd^2)` time and
        inverting it takes :math:`\mathcal{O}(d^3)` time, where :math:`n` is the size
        of the training dataset.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            check_eigvals: bool = False
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]
        hess = 0.0

        for batch, batch_size in self._loader_wrapper(train=True):
            def f(theta_):
                self._model_reinsert_params(self._reshape_like_params(theta_))
                return self.objective.train_loss(self.model, theta_, batch)

            hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
            hess = hess + hess_batch * batch_size

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
            hess = hess / len(self.train_loader.dataset)
            hess = hess + damp * torch.eye(d, device=hess.device)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()

            self.inverse_hess = torch.inverse(hess)

    def inverse_hvp(self, vec):
        return self.inverse_hess @ vec


class CGInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the method of (truncated) Conjugate Gradients (CG).

    This module relies :func:`scipy.sparse.linalg.cg()` to perform CG.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        **kwargs: keyword arguments which are passed into the "Other Parameters" of
            :func:`scipy.sparse.linalg.cg()`.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            gnh: bool = False,
            **kwargs
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.cg_kwargs = kwargs

    def inverse_hvp(self, vec):
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        def hvp_fn(v):
            v = torch.tensor(v, requires_grad=False, device=self.device, dtype=vec.dtype)

            hvp = 0.0
            for batch, batch_size in self._loader_wrapper(train=True):
                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=v, gnh=self.gnh)
                hvp = hvp + hvp_batch.detach() * batch_size
            hvp = hvp / len(self.train_loader.dataset)
            hvp = hvp + self.damp * v

            return hvp.cpu().numpy()

        d = vec.shape[0]
        linop = L.LinearOperator((d, d), matvec=hvp_fn)
        ihvp = L.cg(A=linop, b=vec.cpu().numpy(), **self.cg_kwargs)[0]

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return torch.tensor(ihvp, device=self.device)


class LiSSAInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the Linear time Stochastic Second-Order Algorithm (LiSSA).

    At a high level, LiSSA estimates an inverse-Hessian vector product
    by using truncated Neumann iterations:

    .. math::
        \mathbf{H}^{-1}\mathbf{v} \approx \frac{1}{R}\sum\limits_{r = 1}^R
        \left(\sigma^{-1}\sum_{t = 1}^{T}(\mathbf{I} - \sigma^{-1}\mathbf{H}_{r, t})^t\mathbf{v}\right)

    Here, :math:`\mathbf{H}` is the risk Hessian matrix and :math:`\mathbf{H}_{r, t}` are
    loss Hessian matrices over batches of training data drawn randomly with replacement (we
    also use a batch size in ``train_loader``). In addition, :math:`\sigma > 0` is a scaling
    factor chosen sufficiently large such that :math:`\sigma^{-1} \mathbf{H} \preceq \mathbf{I}`.

    In practice, we can compute each inner sum recursively. Starting with
    :math:`\mathbf{h}_{r, 0} = \mathbf{v}`, we can iteratively update for :math:`T` steps:

    .. math::
        \mathbf{h}_{r, t} = \mathbf{v} + \mathbf{h}_{r, t - 1} - \sigma^{-1}\mathbf{H}_{r, t}\mathbf{h}_{r, t - 1}

    where :math:`\mathbf{h}_{r, T}` will be equal to the :math:`r`-th inner sum.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        repeat: the number of trials :math:`R`.
        depth: the recurrence depth :math:`T`.
        scale: the scaling factor :math:`\sigma`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        debug_callback: a callback function which is passed in :math:`(r, t, \mathbf{h}_{r, t})`
            at each recurrence step.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            repeat: int,
            depth: int,
            scale: float,
            gnh: bool = False,
            debug_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            graph: Optional[torch_geometric.data.Data] = None,
            full_batch: bool = False,
            lissa_iter: int = 5000
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.repeat = repeat
        self.depth = depth
        self.scale = scale
        self.debug_callback = debug_callback
        self.graph = graph
        self.full_batch = full_batch
        self.lissa_iter = lissa_iter

    def _increase_scale(self, mode: str):
        prev_scale = float(self.scale)
        if not math.isfinite(prev_scale) or prev_scale <= 0:
            prev_scale = 1.0

        if mode in {"nonfinite_hvp", "nan_norm", "nonfinite_h_est"}:
            if prev_scale < 64:
                next_scale = prev_scale * 2
            elif prev_scale < 512:
                next_scale = prev_scale + 64
            else:
                next_scale = prev_scale + 128
        elif mode == "divergence":
            if prev_scale < 64:
                next_scale = prev_scale * 2
            elif prev_scale < 1024:
                next_scale = prev_scale * 1.5
            else:
                next_scale = prev_scale + 128
        else:
            next_scale = prev_scale * 2

        self.scale = min(float(next_scale), 1e6)
        return prev_scale, float(self.scale)

    def inverse_hvp(self, vec):

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        ihvp = 0.0

        for r in range(self.repeat):

            h_est = vec.clone()

            for t, (batch, _) in enumerate(self._loader_wrapper(sample_n_batches=self.depth, train=True)):

                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=h_est, gnh=self.gnh)

                with torch.no_grad():
                    hvp_batch = hvp_batch + self.damp * h_est
                    h_est = vec + h_est - hvp_batch / self.scale

                if self.debug_callback is not None:
                    self.debug_callback(r, t, h_est)

            ihvp = ihvp + h_est / self.scale

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return ihvp / self.repeat
    
    def inverse_hvp_on_graph(self, vec):
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        r = 0
        restart_cnt = 0
        max_restarts = 20
        h_est = vec.clone()
        prev_h_est = None
        norm = torch.tensor(float("inf"))

        while r < self.lissa_iter - 1:
            hvp = self._hvp_graph(self.graph, flat_params, vec=h_est, gnh=self.gnh)

            if not torch.isfinite(hvp).all():
                restart_cnt += 1
                prev_scale, next_scale = self._increase_scale("nonfinite_hvp")
                print(
                    f"Warning: non-finite HVP at iter {r}. Restart {restart_cnt}/{max_restarts}. "
                    f"Scale {prev_scale:.2f} -> {next_scale:.2f}"
                )
                h_est = vec.clone()
                prev_h_est = None
                r = 0
                if restart_cnt >= max_restarts:
                    raise RuntimeError("LiSSA failed: HVP produced non-finite values too many times.")
                continue

            with torch.no_grad():
                hvp = hvp + self.damp * h_est
                h_est = vec + h_est - hvp / self.scale

            if not torch.isfinite(h_est).all():
                restart_cnt += 1
                prev_scale, next_scale = self._increase_scale("nonfinite_h_est")
                print(
                    f"Warning: non-finite recursion state at iter {r}. Restart {restart_cnt}/{max_restarts}. "
                    f"Scale {prev_scale:.2f} -> {next_scale:.2f}"
                )
                h_est = vec.clone()
                prev_h_est = None
                r = 0
                if restart_cnt >= max_restarts:
                    raise RuntimeError("LiSSA failed: recurrent estimate became non-finite too many times.")
                continue

            if r >= 1:
                norm = torch.norm(h_est - prev_h_est)
                norm_value = float(norm.detach().cpu())

                if not math.isfinite(norm_value):
                    restart_cnt += 1
                    prev_scale, next_scale = self._increase_scale("nan_norm")
                    print(
                        f"Warning: non-finite norm at iter {r}. Restart {restart_cnt}/{max_restarts}. "
                        f"Scale {prev_scale:.2f} -> {next_scale:.2f}"
                    )
                    h_est = vec.clone()
                    prev_h_est = None
                    r = 0
                    if restart_cnt >= max_restarts:
                        raise RuntimeError("LiSSA failed: repeated NaNs in recursions.")
                    continue
                if self.gnh and r > 100 and norm_value > 10 and self.scale < 50000:
                    prev_scale, next_scale = self._increase_scale("divergence")
                    print(
                        f"Warning: Divergence detected at iteration {r}. Norm: {norm_value}. "
                        f"Scale {prev_scale:.2f} -> {next_scale:.2f}"
                    )
                    h_est = vec.clone()
                    prev_h_est = None
                    r = 0
                    continue
                if not self.gnh and r > 10 and norm_value > 10 and self.scale < 100000:
                    prev_scale, next_scale = self._increase_scale("divergence")
                    print(
                        f"Warning: Divergence detected at iteration {r}. Norm: {norm_value}. "
                        f"Scale {prev_scale:.2f} -> {next_scale:.2f}"
                    )
                    h_est = vec.clone()
                    prev_h_est = None
                    r = 0
                    continue
                if norm_value < 1e-7:
                    print(f"LiSSA converged. Norm: {norm_value:.7f}")
                    break
            r += 1
            prev_h_est = h_est.clone()

            # Reduce per-iteration stdout overhead; frequent writes become a bottleneck
            # when many multirun jobs execute LiSSA in parallel.
            if r <= 1 or (r % 25 == 0) or (r >= self.lissa_iter - 2):
                if r <= 1:
                    display_progress(f"Calc. inverse_hvp recursions. ", r, self.lissa_iter)
                else:
                    display_progress(f"Calc. inverse_hvp recursions. Norm: {norm_value:4f}", r, self.lissa_iter)
        ihvp = h_est / self.scale

        if not torch.isfinite(norm):
            raise RuntimeError("LiSSA failed to converge: final norm is NaN.")

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
        print(f'Norm: {norm:.6f}')

        return ihvp, norm
