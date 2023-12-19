class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def add_session_engagement_features(self):
        self.data['LevelsPerSession'] = self.data['max_lvl_no'] / self.data['session_cnt']
        self.data['InteractivityPerSession'] = (self.data['hint1_cnt'] + self.data['hint2_cnt'] + self.data[
            'hint3_cnt'] + self.data['bonus_cnt'] + self.data['repeat_cnt']) / self.data['session_cnt']
        self.data['AvgGameplayDurationPerSession'] = self.data['gameplay_duration'] / self.data['session_cnt']

    def add_player_efficiency_and_success_features(self):
        # Calculate positive aspects of gameplay, including gold claims
        self.data['PositiveGameplay'] = (
                self.data['max_lvl_no'] + self.data['gameplay_duration'] + self.data['claim_gold_cnt'])

        # Calculate penalties based on hints, repeats, and bonuses normalized by positive gameplay aspects
        self.data['Penalty'] = (self.data['hint1_cnt'] + self.data['hint2_cnt'] +
                                self.data['hint3_cnt'] + self.data['repeat_cnt'] +
                                self.data['bonus_cnt']) / self.data['PositiveGameplay']

        self.data['PenaltyInteractivity'] = (self.data['hint1_cnt'] + self.data['hint2_cnt'] +
                                             self.data['hint3_cnt'] + self.data['repeat_cnt'] +
                                             self.data['bonus_cnt'])

        # Calculate Game Efficiency Rate by considering positive gameplay and subtracting the penalty
        self.data['GameEfficiencyRate'] = self.data['PositiveGameplay'] - self.data['Penalty']

    def add_ad_interaction_features(self, weights):
        self.data['WeightedAdInteraction'] = sum(self.data[ad] * weight for ad, weight in weights.items())
        self.data['AdInteractionPerSession'] = (self.data['banner_cnt'] + self.data['is_cnt'] + self.data['rv_cnt']) / \
                                               self.data['session_cnt']

    def execute_all(self, weights):
        self.add_session_engagement_features()
        self.add_player_efficiency_and_success_features()
        self.add_ad_interaction_features(weights)
        return self.data
