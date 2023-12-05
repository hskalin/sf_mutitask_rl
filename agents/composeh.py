from compose import CompositionAgent

class FrameStackCompositionAgent(CompositionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.framestacked_replay = self.buffer_cfg["framestacked_replay"]
        

