class BaseMetric(object):
    def __call__(self, generated_path, ref_path, mun_samples, batch):
        raise NotImplementedError