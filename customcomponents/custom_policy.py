import logging, pprint

from typing import Any, List, Dict, Text, Optional, Set, Tuple, TYPE_CHECKING, Union


import ast, re

from rasa.shared.constants import DOCS_URL_RULES, INTENT_MESSAGE_PREFIX
from rasa.shared.exceptions import RasaException

from rasa.shared.core.events import (
    BotUttered,
    SlotSet,
    UserUttered,
    ActionExecuted,
    ActionReverted,
    UserUtteranceReverted,
    # Event,
)

# from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter

# from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData, PolicyPrediction, Policy
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_active_loop_name,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import DEFAULT_CORE_FALLBACK_THRESHOLD, RULE_POLICY_PRIORITY
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
)
from rasa.shared.core.domain import InvalidDomain, State, Domain
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT,
    TEXT,
)

# import rasa.core.test
# import rasa.core.training.training

if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble

logger = logging.getLogger(__name__)


def extract_b_i_e_t(
    bot_utterance: BotUttered, user_utterance: UserUttered
) -> Tuple[list, dict, dict, str, bool]:
    bot_utterance_dict = bot_utterance.as_dict()
    buttons = bot_utterance_dict["data"].get("buttons")
    disabled = not (not bot_utterance_dict["data"].get("button_intents_disabled", False))  # True'ish
    if not disabled:
        user = user_utterance.as_dict()

        pdata = user.get("parse_data", {})  # parse data has been checked upfront!

        intent = pdata.get(INTENT, {})
        text = pdata.get(TEXT, "")
        entities = {
            e[ENTITY_ATTRIBUTE_TYPE].lower(): (
                e[ENTITY_ATTRIBUTE_VALUE].lower()
                if isinstance(e[ENTITY_ATTRIBUTE_VALUE], str)
                else e[ENTITY_ATTRIBUTE_VALUE]
            )
            for e in pdata.get(ENTITIES, [])
        }

        return buttons, intent, entities, text, disabled
    else:
        logger.debug("Button Intents are disabled")
        return [], {}, {}, "", True


class ButtonPolicy(Policy):
    def __init__(
        self,
        priority: int = RULE_POLICY_PRIORITY + 1,
        button_action_name: str = "action_process_button_answer",
        **kwargs,
    ):
        super().__init__(
            priority=priority,
            **kwargs,
        )
        self.priority = priority
        self.button_action_name: str = button_action_name
        self.params = {}
        for (k, i) in kwargs.items():
            self.params[k] = i
        pass

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:

        return None

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def _prediction_result(
        self, action_name: Text, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Generates a list of actions with probabilities - the probability for the action_name will be set to 1.0 or NLU confidence respectively.

        Args:
            action_name (Text): name of the action to set
            tracker (DialogueStateTracker): Tracker object
            domain (Domain): Domain object

        Returns:
            List[float]: [description]
        """
        logger.debug(f"enter _predict_result")
        result = self._default_predictions(domain)  # list of zeroes in length of actions.
        if action_name:

            score = 1.0

            result[domain.index_for_action(action_name)] = score
        logger.debug(f"exit _predict_result")
        return result

    def _predict_nothing(self, tracker: DialogueStateTracker, domain: Domain) -> PolicyPrediction:
        """Returns a PolicyPrediction with 0.0 Confidence for all registered actions in the Domain

        Args:
            tracker (DialogueStateTracker): [description]
            domain (Domain): [description]

        Returns:
            PolicyPrediction: [description]
        """
        logger.debug(f"_predict_nothing")
        return PolicyPrediction(
            probabilities=self._default_predictions(domain),
            policy_name=self.__class__.__name__,
            policy_priority=self.priority,  # top priority
            events=[],
            optional_events=[],
        )

    def _predict_button_action(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> PolicyPrediction:
        """Returns a PolicyPrediction for the action_noop (name configurable in Policy)

        Args:
            tracker (DialogueStateTracker): Current Dialog's Tracker objeuct
            domain (Domain): Domain object

        Returns:
            PolicyPrediction: A prediction with the button action with confidence = 1.0
        """
        logger.debug(f"enter _predict_button_action")
        return PolicyPrediction(
            probabilities=self._prediction_result(
                action_name=self.button_action_name, tracker=tracker, domain=domain
            ),
            policy_name=self.__class__.__name__,
            policy_priority=self.priority,  # top priority
            events=[],
            optional_events=[],
        )

    def _check_condition_for_button(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[bool, Union[UserUttered, None], Union[BotUttered, None], int]:
        """Check the condition if the uttonPolicy applies here


        Args:
            tracker (DialogueStateTracker)
            domain (Domain)

        Returns:
            Tuple[bool, UserUttered, BotUttered, int]: True, if the condition applies + Last User Utterance and the BotUtterance conaing the buttons
        """
        logger.debug(f"enter _check_condition_for_button")
        # TODO replace hard coded event numbers by filtered events (no $ intents etc)

        applied_events = tracker.applied_events()

        skip = 0
        while isinstance(applied_events[-1 - skip], SlotSet):
            # jump over the slot set (autofill) actions after the NLU
            skip += 1
        logger.debug(f" _check_condition_for_button skip == {skip}")
        if isinstance(applied_events[-1 - skip], UserUttered) and len(applied_events) >= 3:
            logger.debug(" user ok")
            # if user uttered is a literal intent (button click) abort here
            pdata: dict = applied_events[-1 - skip].as_dict().get("parse_data")
            if pdata:
                if pdata.get(TEXT, "").startswith(INTENT_MESSAGE_PREFIX):
                    logger.debug("exit _check_condition_for_button == False, Button is clicked!")
                    return (False, None, None, 0)
            if (
                isinstance(applied_events[-2 - skip], ActionExecuted)
                and applied_events[-2 - skip].as_dict()["name"] == ACTION_LISTEN_NAME
            ):
                logger.debug(" action ok - last action was action_listen")

                if isinstance(applied_events[-3 - skip], BotUttered):
                    logger.debug(" bot ")
                    # check if the last bot utterance before was a button option
                    button_utterance_event = tracker.events[-3 - skip ]
                    button_utterance = button_utterance_event.as_dict()
                    if not button_utterance.get("data") is None:
                        logger.debug(" data ")
                        if not button_utterance["data"].get("buttons") is None:
                            logger.debug(" buttons! ")
                            # we know it was a button question!
                            user_event = applied_events[-1 - skip]
                            user = user_event.as_dict()
                            pdata = user.get("parse_data")
                            if pdata is not None:
                                logger.debug(f"exit _check_condition_for_button == (True,...)")
                                return (
                                    True,
                                    user_event,
                                    button_utterance_event,
                                    skip ,
                                )
        logger.debug("exit _check_condition_for_button == False, Not a button condition!")
        return (False, None, None, 0)

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> "PolicyPrediction":
        """Predicts the next action (see parent class for more information)."""
        # prediction, _ = self._predict(tracker, domain)
        # return prediction

        # Check whether the policy is applicable
        # - the last bot utterance was a button utterance with additional meta data?
        # - the last action was action_listen
        # - the last event was a user utterance
        # execute the rewriting rule
        #  what needs to happen to the events for the bot to run the analysis again?
        #  - probably nothing, as the tracker state is "repaired" and the next best action will be forecasted
        #    based on the tracker state!
        # so the rule needs to repair the tracker itself!
        #

        logger.debug(f"enter predict_action_probabilities")
        logger.debug(f"Tracker check")
        for i, ev in enumerate(tracker.events):
            logger.debug(f"event #{i}")
            logger.debug(pprint.pformat(ev.as_dict(), sort_dicts=False)[:200])
        #logger.debug(tracker.latest_message)
        #logger.debug(pprint.pformat(tracker._latest_message_data(), sort_dicts=False))
        logger.debug("------- APPLIED EVENTS ------")
        for i, ev in enumerate(tracker.applied_events()):
            logger.debug(f"event #{i}")
            logger.debug(pprint.pformat(ev.as_dict(), sort_dicts=False)[:200])
        logger.debug("------- ------------- ------")
        done = False

        # check, skip = self._check_condition_for_button(tracker, domain)
        check, user_utterance_event, button_utterance, skips = self._check_condition_for_button(
            tracker, domain
        )
        if not check:
            return self._predict_nothing(tracker, domain)

        # we know it was a button question!

        buttons, intent, entities, text, disabled = extract_b_i_e_t(
            button_utterance, user_utterance_event
        )
        if disabled:
            return self._predict_nothing(tracker, domain)

        # logger.debug("entities:")
        # logger.debug(entities)

        return self._predict_button_action(tracker, domain)

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "button_action_name": self.button_action_name,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "button_policy.json"
