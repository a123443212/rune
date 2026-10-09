class GameSpec:
    game_id = ""
    num_groups = 0
    vocabs = ()
    tokens = 8
    token_dim = 32
    context_dim = 0
    context_names = ()
    feature_version = ""

    def extract(self, state):
        raise NotImplementedError

    def context(self, state):
        raise NotImplementedError

    def phase(self, state):
        raise NotImplementedError

    def normalize(self, state):
        return state

    def phase_from_ids(self, group_ids, group_mask):
        raise NotImplementedError
