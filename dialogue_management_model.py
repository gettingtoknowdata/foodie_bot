from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import CmdlineInput as ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.featurizers import (MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer)

logger = logging.getLogger(__name__)


	
def run_restaurant_bot(serve_forever=True):
	interpreter = RasaNLUInterpreter('./models/nlu/default/restaurantnlu')
	agent = Agent.load('./models/dialogue', interpreter = interpreter)
	
	if serve_forever:
		agent.handle_channels([ConsoleInputChannel()])
		
	return agent
	
if __name__ == '__main__':
	# train_dialogue()
	run_restaurant_bot()
