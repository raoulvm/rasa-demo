# -*- coding: utf-8 -*-
import logging
import re
import ast
import pprint
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Text, Tuple

""" rasa oss is not installed in OpenShift !
############################################
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT,
    TEXT,
)
"""

INTENT_MESSAGE_PREFIX = "/"
INTENT_NAME_KEY = "name"
PREDICTED_CONFIDENCE_KEY = "confidence"
ENTITIES = "entities"
ENTITY_ATTRIBUTE_TYPE = "entity"
ENTITY_ATTRIBUTE_VALUE = "value"
INTENT = "intent"
TEXT = "text"

from rasa_sdk import Action, Tracker
from rasa_sdk.interfaces import ACTION_LISTEN_NAME
from rasa_sdk.types import DomainDict
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import (
    ActionExecuted,
    ActionExecutionRejected,
    SlotSet,
    UserUtteranceReverted,
    EventType,
    UserUttered,
)


logger = logging.getLogger(__name__)

INTENTFILENAME = r"./config/button_intents.json"


def read_button_intents_defaults(filename: str = INTENTFILENAME) -> dict:
    result = {
        "use_default_intents": False,
        "delete_entities": True,  # delete entities from "inform" or other alternate intents
        "keep_non_influencing_slots": True,
        "intent_inform_ordinal_name": "",
        "max_numerical_intents": 0,
        "intent_inform_left_name": "",
        "intent_inform_right_name": "",
        "intent_inform_last_name": "",
        "intent_inform_middle_name": "",
    }
    try:
        logger.debug(f"Try opening file {filename}")
        with open(file=filename, mode="rt") as f:
            result = json.load(f)
            logger.debug(f"Parsed file {filename} to {result}")
    except Exception as e:
        logger.debug(f"FAILED opening file {filename} with {e}")

    return result


def extract_b_i_e_t(
    bot_utterance: EventType, user_utterance: EventType
) -> Tuple[list, dict, dict, str, bool]:
    """Extracts buttons, intent, entities, text and disabled from the last Bot utterance and the last user utterance

    Args:
        bot_utterance (EventType-BotUttered-Dict): The question asked by the bot that contained buttons
        user_utterance (EventType-UserUttered-Dict): [description]

    Returns:
        Tuple[list, dict, dict, str, bool]: buttons, intent, entities, text and disabled
    """
    data: dict = bot_utterance.get("data", {})
    buttons = data.get("buttons", [])

    for b in buttons:
        if b.get("button_intents_disabled", False):
            logger.debug("buttons policy disabled for this utterance.")
            return [], {}, {}, "", True

    pdata = user_utterance.get("parse_data", {})

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

    return buttons, intent, entities, text, False


class ActionButtonAnswer(Action):
    def __init__(self, settingsfilename: str = INTENTFILENAME) -> None:
        super().__init__()
        defaults = read_button_intents_defaults(filename=settingsfilename)
        self.use_default_intents: bool = defaults.get("use_default_intents", True)
        self.delete_entities: bool = defaults.get("delete_entities", True)
        self.keep_non_influencing_slots = defaults.get("keep_non_influencing_slots", True)
        self.intent_inform_ordinal_name: str = defaults.get("intent_inform_ordinal_name", "")
        self.max_numerical_intents: int = defaults.get("max_numerical_intents", 0)
        self.intent_inform_left_name: str = defaults.get("intent_inform_left_name", "")
        self.intent_inform_right_name: str = defaults.get("intent_inform_right_name", "")
        self.intent_inform_last_name: str = defaults.get("intent_inform_last_name", "")
        self.intent_inform_middle_name: str = defaults.get("intent_inform_middle_name", "")

    def name(self) -> Text:
        return "action_process_button_answer"

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
            if buttonnumber == 0 and self.intent_inform_left_name:
                intents.append(self.intent_inform_left_name)
            if self.intent_inform_ordinal_name and buttonnumber < self.max_numerical_intents:
                intents.append(self.intent_inform_ordinal_name.replace("#", str(buttonnumber + 1)))
            if (
                (buttoncount) % 2 == 1
                and int(buttoncount - 1) / 2 == buttonnumber
                and self.intent_inform_middle_name
            ):  # odd number of buttons and n is the middle
                intents.append(self.intent_inform_middle_name)
            if buttonnumber == buttoncount - 1 and self.intent_inform_last_name:
                intents.append(self.intent_inform_last_name)
            if buttonnumber == buttoncount - 1 and self.intent_inform_right_name:
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
        logger.debug("_is_name_in_intentlist_no_ent ")
        logger.debug(
            f"exit is_name_in_intentlist_no_ent == {intentname in [i for i in intents if isinstance(i, str)]}"
        )
        return intentname in [i for i in intents if isinstance(i, str)]

    def _process_button(
        self,
        buttonnumber: int,
        buttoncount: int,
        intents: list,
        intentname: str,
        entities: dict,
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
        logger.debug("enter _process_button")

        # amend with default button_intents
        intents.extend(self._get_default_intents(buttonnumber, buttoncount))

        req_intent_checklist = [
            list(i.keys())[0] if isinstance(i, dict) else i for i in intents
        ]  # all intents to search for (regardless of entities)

        if self._is_name_in_intentlist_no_ent(intents, intentname):
            logger.debug("exit _process_button == True")
            return True
        if intentname not in req_intent_checklist:
            # intent is not there at all
            logger.debug("exit _process_button == False")
            return False

        # extract entity requirements from button intents that have the same intent name as the intent from NLU
        # TODO fix for strings in the dict, lists are working!!!
        req_intents_with_entities = [
            v
            for i in intents
            if isinstance(i, dict) and list(i.keys())[0] == intentname
            for v in list(i.values())
        ]
        logger.debug("req_intents_with_entities")
        logger.debug(req_intents_with_entities)

        if not req_intents_with_entities:
            # at least one intent with that name requires an entity
            # check all entity values for a match!
            # intents are OR conditions
            # entities are AND conditions
            # entity values are OR conditions
            logger.debug("exit _process_button empty list of required intents")
            return False

        for req_list_of_entities in req_intents_with_entities:
            for req_entity in req_list_of_entities:

                logger.debug(f"req_entity == {req_entity} in {entities} ?")

                if isinstance(req_entity, str) and req_entity in entities:
                    # single entity without value requirement
                    # fit, remove requirement
                    req_list_of_entities.remove(req_entity)
                    logger.debug("FIT FOUND WITH NO VALUE")
                elif isinstance(req_entity, Dict):
                    # entity with one or more value requirements
                    logger.debug("Value required")
                    req_ent_key: str = list(req_entity.keys())[0]
                    if (
                        isinstance(req_entity[req_ent_key], str)
                        and (req_ent_key, req_entity[req_ent_key]) in entities.items()
                    ):
                        # one string as value
                        # format the same as in rasa entity list?! - no: key:value dict, not list of dicts as rasa uses
                        # fit, remove requirement
                        req_list_of_entities.remove(req_entity)
                        logger.debug("FIT FOUND IN SINGLE VALUE")
                        logger.debug(req_list_of_entities)
                    elif isinstance(req_entity[req_ent_key], list):
                        # multiple possible values, iterate
                        logger.debug("iterate multiple values:")
                        for val in req_entity[req_ent_key]:
                            # Value of the first (and only) key
                            logger.debug(f"{req_ent_key}:{val}")
                            if (req_ent_key, val) in entities.items():
                                # fit, remove requirement
                                req_list_of_entities.remove(req_entity)
                                logger.debug("FIT FOUND FROM ITERATE")
                                logger.debug(req_list_of_entities)
                logger.debug("-- end of check --")
            if len(req_list_of_entities) == 0:
                # removed all requirements
                logger.debug("-- SUCCESS --")
                logger.debug("exit _process_button == True")
                return True
        logger.debug("exit _process_button == False (end of loop)")
        return False

    def _create_events(
        self,
        tracker: Tracker,
        dispatcher: CollectingDispatcher,
        button: dict,
        text: str,
        orig_intent: dict,
        user_utterance: EventType,
        original_entities: List[dict] = [],
        original_slots: List[dict] = [],
    ) -> List[EventType]:
        # store the skipped events
        logger.debug("modify tracker")

        # revert the user utterance
        result = [
            UserUtteranceReverted(timestamp=time.time()),
            ActionExecuted(
                ACTION_LISTEN_NAME,
                policy="lied_about_the_policy",
                confidence=1.0,
                timestamp=time.time(),
            ),
        ]

        payload: str = button.get("payload", "")
        if payload.startswith(INTENT_MESSAGE_PREFIX):
            # raise ValueError("Button Payload is no literal intent")
            # remove intent prefix (default "/")
            payload = payload[len(INTENT_MESSAGE_PREFIX) :]

            #  fix intents with buttons as payloads!
            payloadentities = {}
            if "{" in payload:
                # contains entities
                payloadentities = re.findall(r"{.*?}", payload)
                if payloadentities:
                    payloadentities = ast.literal_eval(payloadentities[0])
                else:
                    raise ValueError(f"Failed to parse {button.get('payload')} ")
                payload = payload[: payload.index("{")]  # remove entities from intent name
            entitylist = [
                {
                    ENTITY_ATTRIBUTE_TYPE: k,
                    ENTITY_ATTRIBUTE_VALUE: v,
                    "processors": ["button_policy"],
                }
                for k, v in payloadentities.items()
            ]
            entitylist.extend(original_entities)  # add the original entities if they are passed
            utterance = UserUttered(
                text=text,
                parse_data={
                    # **user_utterance.get("parse_data",{}),
                    INTENT: {
                        INTENT_NAME_KEY: payload,
                        PREDICTED_CONFIDENCE_KEY: orig_intent[PREDICTED_CONFIDENCE_KEY],
                    },
                    ENTITIES: entitylist,  # replace entities
                },
                input_channel=user_utterance.get("input_channel", ""),
                timestamp=time.time(),
            )
        else:
            # it is a clear text utterance in the payload
            logger.info(f'Clear text payload "{payload}"')
            utterance = UserUttered(
                text=payload,
                input_channel=user_utterance.get("input_channel", ""),
                timestamp=time.time(),
            )
        result.append(utterance)

        # re-create the original slot set events for non-influencing slots
        for e in original_slots:
            result.append(SlotSet(key=e.get("name"), value=e.get("value"), timestamp=time.time()))

        result.append(
            ActionExecutionRejected(
                action_name="action_process_button_answer", timestamp=time.time()
            )
        )
        logger.debug("Result returned:")
        logger.debug(pprint.pformat(result))
        return result

    def get_applied_events_for(
        self, tracker: Tracker, event_type: Text, after_timestamp: float, skip: int = 0
    ) -> Iterable:
        def filter_function(e: Dict[Text, Any]) -> bool:
            has_instance = e["event"] == event_type
            excluded = (e.get("timestamp", 0) or 0) <= (after_timestamp or 0)

            return has_instance and not excluded

        filtered = filter(filter_function, reversed(tracker.applied_events()))
        for _ in range(skip):
            next(filtered, None)

        return filtered

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:

        # process the last answer in the light of a button quetstion asked before
        logger.debug("all events ---------------")
        logger.debug(pprint.pformat(tracker.events, sort_dicts=False))
        logger.debug("--------------- APPLIED events ---------------")
        logger.debug(pprint.pformat(tracker.applied_events(), sort_dicts=False))

        # if this run method is called, we know the button policy already has checked
        # the basic conditions.

        # get the last bot an duser utterances for button processing
        user_utterance = tracker.get_last_event_for("user") or {}
        bot_utterance = tracker.get_last_event_for("bot") or {}

        buttons, intent, entities, text, disabled = extract_b_i_e_t(
            bot_utterance=bot_utterance, user_utterance=user_utterance
        )
        if disabled:
            return [
                ActionExecutionRejected(
                    action_name="action_process_button_answer", timestamp=time.time()
                )
            ]

        # store the slot events AFTER the user utterance for slots that have "influence_conversation" set to False:
        user_timestamp = user_utterance.get("timestamp", 1e50)
        slot_events = reversed(
            list(
                self.get_applied_events_for(
                    tracker, event_type="slot", after_timestamp=user_timestamp
                )
            )
        )
        slots_non_inf_conv = []
        for e in slot_events:
            # compare to domain
            """
            {'event': 'slot',
            'timestamp': 1635488096.637538,
            'name': 'topic',
            'value': 'festnetz'},
            """
            name = e.get("name")
            if name:
                sl = domain.get("slots").get(name)
                if sl:
                    if not sl.get("influence_conversation", True):
                        slots_non_inf_conv.append(e)
                        logger.debug(f"influence conversation is false: kept {e} for later use")

        # check for extended meta data "button_intents"
        for n, b in enumerate(buttons):
            logger.debug(f"{n} Button {b}")
            button_intents: list = b.get("button_intents", [])
            if self._process_button(
                buttonnumber=n,
                buttoncount=len(buttons),
                intents=button_intents,
                intentname=intent[INTENT_NAME_KEY],
                entities=entities,
            ):
                # we got a match

                return self._create_events(
                    tracker,
                    dispatcher,
                    button=b,
                    text=text,
                    orig_intent=intent,
                    user_utterance=user_utterance,
                    original_entities=[]
                    if self.delete_entities
                    else user_utterance.get("parse_data", {}).get(ENTITIES, []),
                    original_slots=slots_non_inf_conv,
                )

        return [
            ActionExecutionRejected(
                action_name="action_process_button_answer", timestamp=time.time()
            )
        ]
