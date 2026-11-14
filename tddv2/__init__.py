"""TDD-YOLO v2: fixed CSAF/CAFE modules + DySample, trainable with stock ultralytics."""
import warnings

# the original CBAM_cor uses the deprecated torch.cuda.amp.autocast() inside forward(); keep it unchanged, silence it
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"CUDA is not available or torch_xla is imported.*")

from .register import register

register(verbose=False)

from .modules import CAFE2, CSAF2, DySample  # noqa: E402

__all__ = ("register", "CAFE2", "CSAF2", "DySample")
