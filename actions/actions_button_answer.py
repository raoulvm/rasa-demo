# -*- coding: utf-8 -*-
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Text, Optional, Tuple

from rasa.shared.constants import DOCS_URL_RULES, INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import INTENT_NAME_KEY, PREDICTED_CONFIDENCE_KEY, ENTITIES, ENTITY_ATTRIBUTE_VALUE, ENTITY_ATTRIBUTE_TYPE, INTENT, TEXT

from rasa_sdk import Action, Tracker
from rasa_sdk.types import DomainDict
#from rasa_sdk.forms import FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
    BotUttered, 
    UserUttered,
)

from actions import config
from actions.api import community_events
# from actions.api.algolia import AlgoliaAPI
# from actions.api.discourse import DiscourseAPI
# from actions.api.gdrive_service import GDriveService
# from actions.api.mailchimp import MailChimpAPI
# from actions.api.rasaxapi import RasaXAPI

USER_INTENT_OUT_OF_SCOPE = "out_of_scope"

logger = logging.getLogger(__name__)


# defaults ==> make that read a file
use_default_intents: bool = True,
intent_inform_ordinal_name: str = "inform_#_ordinal"
max_numerical_intents: int = 6
intent_inform_left_name: str = "inform_links"
intent_inform_right_name: str = "inform_rechts"
intent_inform_last_name: str = "inform_letzte"
intent_inform_middle_name: str = "inform_mitte"

def extract_b_i_e_t(
    bot_utterance: BotUttered, user_utterance: UserUttered # TODO these are dicts in SDK
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

class ActionButtonAnswer(Action):

    def __init__(self) -> None:
        super().__init__()
        self.use_default_intents: bool = use_default_intents
        self.intent_inform_ordinal_name: str = intent_inform_ordinal_name
        self.max_numerical_intents: int = max_numerical_intents
        self.intent_inform_left_name: str = intent_inform_left_name
        self.intent_inform_right_name: str = intent_inform_right_name
        self.intent_inform_last_name: str = intent_inform_last_name
        self.intent_inform_middle_name: str = intent_inform_middle_name

    def name(self) -> Text:
        return "action_process_button_answer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:

        # process the last answer in the light of a button quetstion asked before

        return []

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

    def _modify_tracker(self, ):
        pass
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
                        payloadentities = re.findall(r"{.*?}", payload)
                        if payloadentities:
                            payloadentities = ast.literal_eval(payloadentities[0])
                        else:
                            raise RasaException(f"Failed to parse {b.get('payload')} ")
                        payload = payload[:payload.index('{')] # remove entities from intent name
                    entitylist = []
                    for k, v in payloadentities.items():
                        entitylist.append({ENTITY_ATTRIBUTE_TYPE: k, ENTITY_ATTRIBUTE_VALUE: v, 'processors':['button_policy']})

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
