from model.module.interaction import BinaryClassInteraction
from model.fm import build_fm
from model.deepfm import build_deepfm
from model.xdeepfm import build_xdeepfm

from utils.log import get_logger


logger = get_logger()

# TODO: Design a arch which consists of linear, deep and interaction part, and fm/deepfm/xdeepfm are all its instance.
MODELS = {
    'fm': build_fm,
    'deepfm': build_deepfm,
    'xdeepfm': build_xdeepfm
}


def build_model(model_cfg):
    logger.info(f'Building model...')
    if model_cfg.name in MODELS.keys():
        model = MODELS[model_cfg.name](model_cfg)

        if getattr(model_cfg, 'binary_class_interaction', False):
            model = BinaryClassInteraction(model_cfg.output_dim, model)

        logger.info(f'Model \"{type(model).__name__}\" is built:\n'
                    f'{model}')
        return model
    else:
        raise ValueError(f'model: {model_cfg.name} is not found.')
