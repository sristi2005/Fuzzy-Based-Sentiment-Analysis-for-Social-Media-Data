import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class FuzzySentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        # Antecedents (Inputs)
        # VADER compound score ranges from -1.0 to 1.0
        vader_score = ctrl.Antecedent(np.arange(-1.0, 1.01, 0.01), 'vader_score')
        # TextBlob polarity ranges from -1.0 to 1.0
        textblob_score = ctrl.Antecedent(np.arange(-1.0, 1.01, 0.01), 'textblob_score')
        
        # Consequent (Output)
        # Final sentiment score ranges from 0 to 100
        final_sentiment = ctrl.Consequent(np.arange(0, 101, 1), 'final_sentiment')

        # Membership functions for VADER
        vader_score['negative'] = fuzz.trapmf(vader_score.universe, [-1.0, -1.0, -0.2, 0.0])
        vader_score['neutral'] = fuzz.trimf(vader_score.universe, [-0.2, 0.0, 0.2])
        vader_score['positive'] = fuzz.trapmf(vader_score.universe, [0.0, 0.2, 1.0, 1.0])

        # Membership functions for TextBlob
        textblob_score['negative'] = fuzz.trapmf(textblob_score.universe, [-1.0, -1.0, -0.2, 0.0])
        textblob_score['neutral'] = fuzz.trimf(textblob_score.universe, [-0.2, 0.0, 0.2])
        textblob_score['positive'] = fuzz.trapmf(textblob_score.universe, [0.0, 0.2, 1.0, 1.0])

        # Membership functions for Final Sentiment
        final_sentiment['very_negative'] = fuzz.trapmf(final_sentiment.universe, [0, 0, 15, 30])
        final_sentiment['negative'] = fuzz.trimf(final_sentiment.universe, [15, 30, 45])
        final_sentiment['neutral'] = fuzz.trimf(final_sentiment.universe, [35, 50, 65])
        final_sentiment['positive'] = fuzz.trimf(final_sentiment.universe, [55, 70, 85])
        final_sentiment['very_positive'] = fuzz.trapmf(final_sentiment.universe, [70, 85, 100, 100])

        # Fuzzy Rules
        rule1 = ctrl.Rule(vader_score['negative'] & textblob_score['negative'], final_sentiment['very_negative'])
        rule2 = ctrl.Rule(vader_score['negative'] & textblob_score['neutral'], final_sentiment['negative'])
        rule3 = ctrl.Rule(vader_score['neutral'] & textblob_score['negative'], final_sentiment['negative'])
        rule4 = ctrl.Rule(vader_score['neutral'] & textblob_score['neutral'], final_sentiment['neutral'])
        rule5 = ctrl.Rule(vader_score['positive'] & textblob_score['neutral'], final_sentiment['positive'])
        rule6 = ctrl.Rule(vader_score['neutral'] & textblob_score['positive'], final_sentiment['positive'])
        rule7 = ctrl.Rule(vader_score['positive'] & textblob_score['positive'], final_sentiment['very_positive'])
        
        # Conflict resolution rules
        rule8 = ctrl.Rule(vader_score['negative'] & textblob_score['positive'], final_sentiment['neutral'])
        rule9 = ctrl.Rule(vader_score['positive'] & textblob_score['negative'], final_sentiment['neutral'])

        # Create control system
        self.sentiment_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9
        ])
        
        self.sentiment_sim = ctrl.ControlSystemSimulation(self.sentiment_ctrl)

    def analyze_text(self, text: str):
        # 1. Get VADER score
        vader_result = self.vader_analyzer.polarity_scores(text)
        v_compound = vader_result['compound']

        # 2. Get TextBlob score
        blob = TextBlob(text)
        t_polarity = blob.sentiment.polarity
        t_subjectivity = blob.sentiment.subjectivity

        # 3. Apply Fuzzy Logic
        self.sentiment_sim.input['vader_score'] = v_compound
        self.sentiment_sim.input['textblob_score'] = t_polarity
        
        try:
            self.sentiment_sim.compute()
            final_score = self.sentiment_sim.output['final_sentiment']
        except ValueError:
            # Fallback if computation fails
            final_score = 50.0

        # Determine textual label
        label = self._get_label_from_score(final_score)

        return {
            "text": text,
            "vader": {
                "compound": v_compound,
                "pos": vader_result['pos'],
                "neu": vader_result['neu'],
                "neg": vader_result['neg']
            },
            "textblob": {
                "polarity": t_polarity,
                "subjectivity": t_subjectivity
            },
            "fuzzy_result": {
                "score": round(final_score, 2),
                "label": label
            }
        }

    def _get_label_from_score(self, score: float) -> str:
        if score < 20:
            return "Very Negative"
        elif score < 40:
            return "Negative"
        elif score < 60:
            return "Neutral"
        elif score < 80:
            return "Positive"
        else:
            return "Very Positive"

# Initialize a global analyzer to be used by the API
analyzer = FuzzySentimentAnalyzer()
