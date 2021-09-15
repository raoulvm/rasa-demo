# -*- coding: utf-8 -*-
import logging, re, ast, pprint
import json
import time
from typing import Any, Dict, List, Text, Optional, Tuple

from rasa.shared.constants import DOCS_URL_RULES, INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT,
    TEXT,
)

from rasa_sdk import Action, Tracker
from rasa_sdk.interfaces import ACTION_LISTEN_NAME
from rasa_sdk.types import DomainDict

# from rasa_sdk.forms import FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import (
    ActionExecuted,
    SlotSet,
    UserUtteranceReverted,
    ActionReverted,
    ConversationPaused,
    EventType,
    BotUttered,
    UserUttered,
)

from actions import config


# USER_INTENT_OUT_OF_SCOPE = "out_of_scope"

logger = logging.getLogger(__name__)


# defaults ==> make that read a file
use_default_intents: bool = True
delete_entities: bool = True  # delete entities from "inform" or other alternate intents

intent_inform_ordinal_name: str = "inform_#_ordinal"
max_numerical_intents: int = 6
intent_inform_left_name: str = "inform_links"
intent_inform_right_name: str = "inform_rechts"
intent_inform_last_name: str = "inform_letzte"
intent_inform_middle_name: str = "inform_mitte"


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
    disabled = not (not data.get("button_intents_disabled", False))  # True'ish
    if not disabled:

        pdata = user_utterance.get("parse_data", {})  # parse data has been checked upfront!

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


class ActionButtonAnswer(Action):
    def __init__(self) -> None:
        super().__init__()
        self.use_default_intents: bool = use_default_intents
        self.delete_entities: bool = delete_entities
        self.intent_inform_ordinal_name: str = intent_inform_ordinal_name
        self.max_numerical_intents: int = max_numerical_intents
        self.intent_inform_left_name: str = intent_inform_left_name
        self.intent_inform_right_name: str = intent_inform_right_name
        self.intent_inform_last_name: str = intent_inform_last_name
        self.intent_inform_middle_name: str = intent_inform_middle_name

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
        self, buttonnumber: int, buttoncount: int, intents: list, intentname: str, entities: dict
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
                            if (
                                req_entity[ENTITY_ATTRIBUTE_TYPE],
                                req_entity[ENTITY_ATTRIBUTE_VALUE],
                            ) in entities.items():
                                # fit, remove requirement
                                req_list_of_entities.remove(req_entity)
                                logger.debug("FIT SINGLE VALUE")
                                logger.debug(req_list_of_entities)
                        elif isinstance(list(req_entity.values())[0], list):
                            # multiple possible values, iterate
                            logger.debug("iterate multiple values:")
                            for val in list(req_entity.values())[
                                0
                            ]:  # Value of the first (and only) key
                                logger.debug(f"{req_ent_key}:{val}")
                                if (req_ent_key, val) in entities.items():
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

    def _create_events(
        self,
        tracker: Tracker,
        dispatcher: CollectingDispatcher,
        button: dict,
        text: str,
        orig_intent: dict,
        user_utterance: EventType,
    ) -> List[EventType]:
        # store the skipped events
        logger.debug(f"modify tracker")

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
        if not payload.startswith(INTENT_MESSAGE_PREFIX):
            raise ValueError("Button Payload is no literal intent")
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
        entitylist = []
        for k, v in payloadentities.items():
            entitylist.append(
                {
                    ENTITY_ATTRIBUTE_TYPE: k,
                    ENTITY_ATTRIBUTE_VALUE: v,
                    "processors": ["button_policy"],
                }
            )
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
        result.append(utterance)
        # tracker.latest_message = (
        #     utterance  #  change also last message data at `tracker.latest_message`
        # )
        return result

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
            return []

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
                # TODO create events and return them
                return self._create_events(
                    tracker,
                    dispatcher,
                    button=b,
                    text=text,
                    orig_intent=intent,
                    user_utterance=user_utterance,
                )

        return []
