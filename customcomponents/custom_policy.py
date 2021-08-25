import logging

from typing import Any, List, Dict, Text, Optional, Set, Tuple, TYPE_CHECKING


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
#from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
#from rasa.core.policies.memoization import MemoizationPolicy
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
from rasa.shared.nlu.constants import INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY, ENTITIES, ENTITY_ATTRIBUTE_VALUE, ENTITY_ATTRIBUTE_TYPE, INTENT, TEXT
#import rasa.core.test
#import rasa.core.training.training

if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble

logger = logging.getLogger(__name__)


def extract_b_i_e_t(
    bot_utterance: BotUttered, user_utterance: UserUttered
) -> Tuple[list, object, list, str, bool]:
    t_3 = bot_utterance.as_dict()
    buttons = t_3["data"].get("buttons")
    disabled = not (not t_3["data"].get("button_intents_disabled", False))  # True'ish
    if not disabled:
        user = user_utterance.as_dict()

        pdata = user.get("parse_data")  # parse data has been checked upfront!

        intent = pdata.get(INTENT)
        text = pdata.get(TEXT)
        entities = [
            {
                e[ENTITY_ATTRIBUTE_TYPE].lower(): (
                    e[ENTITY_ATTRIBUTE_VALUE].lower() if isinstance(e[ENTITY_ATTRIBUTE_VALUE], str) else e[ENTITY_ATTRIBUTE_VALUE]
                )
                for e in pdata.get(ENTITIES)
            }
        ]
        return buttons, intent, entities, text, disabled
    else:
        logger.debug("Button Intents are disabled")
        return [], [], [], "", True


class ButtonPolicy(Policy):
    def __init__(
        self,
        priority: int = RULE_POLICY_PRIORITY + 1,
        delete_entities: bool = True,  # delete entities from "inform" or other alternate intents
        execute_noop_action: bool = False,
        noop_action_name: str = "action_noop",
        use_default_intents: bool = True,
        intent_inform_ordinal_name: str = "inform_#_ordinal",
        max_numerical_intents: int = 6,
        intent_inform_left_name: str = "inform_links",
        intent_inform_right_name: str = "inform_rechts",
        intent_inform_last_name: str = "inform_letzte",
        intent_inform_middle_name: str = "inform_mitte",
        **kwargs,
    ):
        super().__init__(
            priority=priority, **kwargs,
        )
        self.priority = priority
        self.delete_entities: bool = delete_entities
        self.execute_noop_action: bool = execute_noop_action
        self.noop_action_name: str = noop_action_name
        self.use_default_intents: bool = use_default_intents
        self.intent_inform_ordinal_name: str = intent_inform_ordinal_name
        self.max_numerical_intents: int = max_numerical_intents
        self.intent_inform_left_name: str = intent_inform_left_name
        self.intent_inform_right_name: str = intent_inform_right_name
        self.intent_inform_last_name: str = intent_inform_last_name
        self.intent_inform_middle_name: str = intent_inform_middle_name
        self.params = {}
        for (k, i) in kwargs.items():
            self.params[k] = i
        pass

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:

        # TODO check intents against domain! - raise exception if wrong

        # TODO check actions against domain! - raise exception if wrong

        logger.debug(f"executed validate_against_domain")
        return None

        if False: return InvalidDomain() # TODO enable that if intents not in domain

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

    def _predict_noop(self, tracker: DialogueStateTracker, domain: Domain) -> PolicyPrediction:
        """Returns a PolicyPrediction for the action_noop (name configurable in Policy)

        Args:
            tracker (DialogueStateTracker): Current Dialog's Tracker objeuct
            domain (Domain): Domain object

        Returns:
            PolicyPrediction: A prediction with the nnop_action with confidence = 1.0
        """
        logger.debug(f"enter _predict_noop")
        return PolicyPrediction(
            probabilities=self._prediction_result(
                action_name=self.noop_action_name, tracker=tracker, domain=domain
            ),
            policy_name=self.__class__.__name__,
            policy_priority=self.priority,  # top priority
            events=[],
            optional_events=[],
        )

    def _check_condition_for_button2(self, tracker, domain) -> Tuple[bool, UserUttered, BotUttered]:
        """Check the condition if the uttonPolicy applies here


        Args:
            tracker ([type]): [description]
            domain ([type]): [description]

        Returns:
            Tuple[bool, UserUttered, BotUttered]: True, if the condition applies + Last User Utterance and the BotUtterance conaing the buttons
        """
        logger.debug(f"enter _check_condition_for_button")
        # TODO replace hard coded event numbers by filtered events (no $ intents etc)

        skip = 0
        while isinstance(tracker.events[-1 - skip], SlotSet):
            # jump over the slot set (autofill) actions after the NLU
            skip += 1
        logger.debug(f" _check_condition_for_button skip == {skip}")
        if isinstance(tracker.events[-1 - skip], UserUttered) and len(tracker.events) >= 3:
            logger.debug(" user ok")
            # if user uttered is a literal intent (button click) abort here
            pdata: dict = tracker.events[-1 - skip].as_dict().get("parse_data")
            if pdata:
                if pdata.get(TEXT, "").startswith(INTENT_MESSAGE_PREFIX):
                    logger.debug("exit _check_condition_for_button == False, Button is clicked!")
                    return (False, None, None, 0)
            if (
                isinstance(tracker.events[-2 - skip], ActionExecuted)
                and tracker.events[-2 - skip].as_dict()['name'] == ACTION_LISTEN_NAME
            ):
                logger.debug(" action ok - last action was action_listen")

                skip2 = 0
                if isinstance(tracker.events[-3 - skip], UserUtteranceReverted):
                    # check for rewind events and skip them!!

                    # Events für NLU fallback (rückwärts)
                    # event: rewind;
                    # event: bot utter_action:utter_ask_rephrase
                    # event: action name: action_default_fallback
                    # event: user_featurization
                    # event: user (unverständliche eingabe) UserUttered
                    while not isinstance(tracker.events[-3 - skip - skip2], UserUttered):
                        skip2 += 1
                    # also skip the user utterance + UserUtteranceReverted
                    skip2 += 2

                if isinstance(tracker.events[-3 - skip - skip2], BotUttered):
                    logger.debug(" bot ")
                    # check if the last bot utterance before was a button option
                    button_utterance_event = tracker.events[-3 - skip - skip2]
                    button_utterance = button_utterance_event.as_dict()
                    if not button_utterance.get("data") is None:
                        logger.debug(" data ")
                        if not button_utterance["data"].get("buttons") is None:
                            logger.debug(" buttons! ")
                            # we know it was a button question!
                            user = tracker.events[-1 - skip].as_dict()
                            pdata = user.get("parse_data")
                            if pdata is not None:
                                logger.debug(f"exit _check_condition_for_button == (True,...)")
                                return (
                                    True,
                                    tracker.events[-1 - skip],
                                    button_utterance_event,
                                    skip + skip2,
                                )
        logger.debug("exit _check_condition_for_button == False, Not a button condition!")
        return (False, None, None, 0)

    def _get_default_intents(self, buttonnumber: int, buttoncount: int) -> list:
        """returns the default intents for a given button number, such as first, middle, last, etc.

        Args:
            buttonnumber (int): Ordinal number of the button (0..buttoncount-1)
            buttoncount (int): Total number of buttons

        Returns:
            list: List of intent names List[str]
        """
        intents = []
        if self.use_default_intents:
            if buttonnumber == 0 and len(self.intent_inform_left_name) > 0:
                intents.append(self.intent_inform_left_name)
            if len(self.intent_inform_ordinal_name) > 0:
                intents.append(self.intent_inform_ordinal_name.replace("#", str(buttonnumber + 1)))
            if (
                (buttoncount) % 2 == 1
                and int(buttoncount - 1) / 2 == buttonnumber
                and len(self.intent_inform_middle_name) > 0
            ):  # odd number of buttons and n is the middle
                intents.append(self.intent_inform_middle_name)
            if buttonnumber == buttoncount - 1 and len(self.intent_inform_last_name) > 0:
                intents.append(self.intent_inform_last_name)
            if buttonnumber == buttoncount - 1 and len(self.intent_inform_right_name) > 0:
                intents.append(self.intent_inform_right_name)
        logger.debug(f"_get_default_intents == {intents})")
        return intents

    def _is_name_in_intentlist_no_ent(self, intents: list, intentname: str) -> bool:
        """Returns True if the intentname is in the intentname is in the list of intents filtered for string types.

        Args:
            intents (list): list of intents, intents are either str or dict types
            intentname (str): literal name of the intent to search for

        Returns:
            bool: 
        """
        # intents with no entity requirements
        logger.debug(f"enter _is_name_in_intentlist_no_ent({intents}, {intentname})")
        logger.debug(f"_is_name_in_intentlist_no_ent ")
        logger.debug(
            f"exit is_name_in_intentlist_no_ent == {intentname in [i for i in intents if isinstance(i, str)]}"
        )
        return intentname in [i for i in intents if isinstance(i, str)]

    def _process_button(
        self, buttonnumber: int, buttoncount: int, intents: list, intentname: str, entities: list
    ) -> bool:
        """Check a button against the current classified intent (and it's entties, if required)

        Args:
            buttonnumber (int): Button number in list (0..buttoncount-1)
            buttoncount (int): total number of buttons
            intents (list): list of intents on the button (button_intents), can contain dict entries for 
                            intents with entity requirements
            intentname (str): name of classified intent
            entities (list): recognized entities in the user utterance

        Returns:
            bool: True if the button is detected using the alternate intents
        """
        logger.debug(f"enter _process_button")

        # amend with default button_intents
        intents.extend(self._get_default_intents(buttonnumber, buttoncount))

        req_intent_checklist = [
            list(i.keys())[0] if isinstance(i, dict) else i for i in intents
        ]  # all intents to search for (regardless of entities)

        if self._is_name_in_intentlist_no_ent(intents, intentname):
            logger.debug(f"exit _process_button == True")
            return True
        if not intentname in req_intent_checklist:
            # intent is not there at all
            logger.debug(f"exit _process_button == False")
            return False

        # extract entity requirements from button intents that have the same intent name as the intent from NLU
        req_intents_with_entities = [
            v
            for i in intents
            if isinstance(i, dict) and list(i.keys())[0] == intentname
            for v in list(i.values())
        ]
        logger.debug("req_intents_with_entities")
        logger.debug(req_intents_with_entities)

        if req_intents_with_entities:
            # at least one intent with that name requires an entity
            # check all entity values for a match!
            # intents are OR conditions
            # entities are AND conditions
            # entity values are OR conditions

            for req_list_of_entities in req_intents_with_entities:
                for req_entity in req_list_of_entities:
                    logger.debug(req_list_of_entities)
                    logger.debug(req_entity)
                    if isinstance(req_entity, str):
                        # single entity without value requirement
                        # just check if it there
                        logger.debug("no value required")
                        if req_entity in entities.keys():
                            logger.debug("FIT NO VALUE")
                            # fit, remove requirement
                            req_list_of_entities.remove(req_entity)
                            logger.debug(req_list_of_entities)
                    elif isinstance(req_entity, Dict):
                        # entity with one or more value requirements
                        logger.debug("Value required!")
                        req_ent_key: str = list(req_entity.keys())[0]

                        if isinstance(list(req_entity.values())[0], str):
                            logger.debug("single value")
                            # one string as value
                            # format the same as in rasa entity list?!
                            if req_entity in entities:
                                # fit, remove requirement
                                req_list_of_entities.remove(req_entity)
                                logger.debug("FIT SINGLE VALUE")
                                logger.debug(req_list_of_entities)
                        elif isinstance(list(req_entity.values())[0], list): 
                            # multiple possible values, iterate
                            logger.debug("iterate multiple values:")
                            for val in list(req_entity.values())[0]:
                                logger.debug(f"{req_ent_key}:{val}")
                                if {req_ent_key: val} in entities:
                                    # fit, remove requirement
                                    req_list_of_entities.remove(req_entity)
                                    logger.debug("FIT FROM ITERATE")
                                    logger.debug(req_list_of_entities)
                    logger.debug("-- end of check --")
                if len(req_list_of_entities) == 0:
                    # removed all requirements
                    logger.debug("-- SUCCESS --")
                    logger.debug(f"exit _process_button == True")
                    return True
        logger.debug(f"exit _process_button == False (end of loop)")
        return False

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

        # Check whether the polcy is applicable
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
            logger.debug(ev.as_dict())
        logger.debug(tracker.latest_message)
        logger.debug(tracker._latest_message_data())
        logger.debug("-------")
        done = False

        # check, skip = self._check_condition_for_button(tracker, domain)
        check, user_utterance_event, button_utterance, skips = self._check_condition_for_button2(
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

        logger.debug("entities:")
        logger.debug(entities)
        # check for extended meta data "button_intents"
        for n, b in enumerate(buttons):
            logger.debug(f"{n} Button {b}")
            button_intents = b.get("button_intents")

            if self.use_default_intents or not button_intents is None:
                ints: list = button_intents or []
                if self._process_button(n, len(buttons), ints, intent[INTENT_NAME_KEY], entities):
                    # found a matching intent!
                    # store the skipped events
                    logger.debug(f"modify tracker")
                    evt_store = [tracker.events.pop() for n in range(skips)]
                    t_1 = tracker.events.pop()
                    if not self.delete_entities:
                        tracker.events.extend(evt_store)  # restore skipped slot events

                    payload: str = b.get("payload")
                    if not payload.startswith(INTENT_MESSAGE_PREFIX):
                        raise RasaException("Button Payload is no literal intent")
                    # remove intent prefix (default "/")
                    payload = payload[len(INTENT_MESSAGE_PREFIX) :]

                    #  fix intents with buttons as payloads!
                    payloadentities = {}
                    if "{" in payload:
                        # contains entities
                        payloadentities = re.findall(r"{({.*?})}", payload)
                        if payloadentities:
                            payloadentities = ast.literal_eval(payloadentities[0])
                        else:
                            raise RasaException(f"Failed to parse {b.get('payload')} ")
                    entitylist = []
                    for k, v in payloadentities.items():
                        entitylist.append({ENTITY_ATTRIBUTE_TYPE: k, ENTITY_ATTRIBUTE_VALUE: v})

                    utterance = UserUttered(
                        text=text,
                        intent={INTENT_NAME_KEY: payload, PREDICTED_CONFIDENCE_KEY: intent[PREDICTED_CONFIDENCE_KEY],},
                        parse_data={
                            **t_1.as_dict(),
                            INTENT_NAME_KEY: payload,
                            PREDICTED_CONFIDENCE_KEY: intent[PREDICTED_CONFIDENCE_KEY],
                            ENTITIES: entitylist,  # replace entities
                        },
                    )
                    tracker.events.append(utterance)
                    tracker.latest_message = (
                        utterance  #  change also last message data at `tracker.latest_message`
                    )
                    done = True
                    # logger.debug("------- AFTER CHANGE --------")
                    # logger.debug(tracker._latest_message_data())
                    # logger.debug("------- END OF CHANGE --------")
                    break

        if self.execute_noop_action and done:

            return self._predict_noop(tracker, domain)

        else:
            # no prediction
            return self._predict_nothing(tracker, domain)

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "delete_entities": self.delete_entities,
            "execute_noop_action": self.execute_noop_action,
            "noop_action_name": self.noop_action_name,
            "use_default_intents": self.use_default_intents,
            "intent_inform_ordinal_name": self.intent_inform_ordinal_name,
            "max_numerical_intents": self.max_numerical_intents,
            "intent_inform_left_name": self.intent_inform_left_name,
            "intent_inform_right_name": self.intent_inform_right_name,
            "intent_inform_last_name": self.intent_inform_last_name,
            "intent_inform_middle_name": self.intent_inform_middle_name,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "test_policy.json"
