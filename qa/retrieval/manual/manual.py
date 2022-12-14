import toml
from config import get_cfg

__all__ = ['Manual']


class Manual:
    __slot__ = ['cfg', 'rule']

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.rule = toml.load(self.cfg.CONTENTUNDERSTANDING.RULE_FILE)

    def get_rule(self, _type='exclusive'):
        return self.rule[_type]


if __name__ == '__main__':
    cfg = get_cfg()
    manual = Manual(cfg)
