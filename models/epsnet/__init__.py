from .mgdm_adapter import mgdm_adapter_pocket


def get_model(config):
    if config.network == 'mgdm_adapter_pocket':
        return mgdm_adapter_pocket(config)

    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
