# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import os
import logging
import datetime
import json
import csv
from typing import Dict, List, Callable, Union
from graph_of_thoughts import controller, operations, prompter, parser
from . import utils


class MalwarePrompter(prompter.Prompter):
    """
    MalwarePrompter provides the generation of prompts for malware analysis

    Inherits from the Prompter class and implements its abstract methods.
    """

    mal_prompt = """<Instruction> Answer the following question. Use provided data sources if required. Output a relevant and coherent answer. </Instruction>

Input: {input}"""

    mal_prompt_cot = """<Instruction> Answer the following question. You can generate any intermediate questions that help to answer the original question, but the final output should be a relevant answer to the original question. </Instruction>

<Approach>
To answer the question follow these steps:
1. Break down the question into more basic, relevant intermediate questions.
2. Answer each of the intermediate questions, using the provided data if required.
3. Using the answers to the smaller questions and with the help of any data required, answer the original question.
</Approach>

<Examples>
Input: What is the file format of a WAV file?
Intermediate questions:
What is a file format?
What is a WAV file?
Answers to intermediate questions:
A file format is a standard way that information is encoded for storage in a computer file. It specifies how bits are used to encode information in a digital storage medium. File formats may be either proprietary or free.
The WAV file is an instance of a Resource Interchange File Format (RIFF) defined by IBM and Microsoft.[3] The RIFF format acts as a wrapper for various audio coding formats.
Output: 
The WAVE file format, being a subset of Microsoft’s RIFF specification, starts with a file header followed by a sequence of data chunks. A WAVE file has a single “WAVE” chunk which consists of two sub-chunks:
- a “fmt” chunk - specifies the data format
- a “data” chunk - contains the actual sample data

Input: What kind of code is provided as a data source?
Intermediate questions:
What is the data source?
Answers to intermediate questions:
Code from the FFmpeg program, which is a collection of libraries and tools to process multimedia content such as audio, video, subtitles and related metadata.
Output: 
C# code.

</Examples>

Input: {input}"""

    tot_improve_prompt = """<Instruction> The following two lists represent an unsorted list of numbers and a sorted variant of that list. The sorted variant is not correct. Fix the sorted variant so that it is correct.
Make sure that the output list is sorted in ascending order, has the same number of elements as the input list ({length}), and contains the same elements as the input list. </Instruction>

<Approach>
To fix the incorrectly sorted list follow these steps:
1. For each number from 0 to 9, compare the frequency of that number in the incorrectly sorted list to the frequency of that number in the input list.
2. Iterate through the incorrectly sorted list and add or remove numbers as needed to make the frequency of each number in the incorrectly sorted list match the frequency of that number in the input list.
</Approach>

<Examples>
Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]
Incorrectly Sorted: [0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 7, 7, 8, 8, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains four extra 0s, two extra 4s and three extra 9s and is missing two 2s.
Output: [0, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9]

Input: [6, 4, 5, 7, 5, 6, 9, 7, 6, 9, 4, 6, 9, 8, 1, 9, 2, 4, 9, 0, 7, 6, 5, 6, 6, 2, 8, 3, 9, 5, 6, 1]
Incorrectly Sorted: [0, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains two extra 4s and is missing two 6s and one 9.
Output: [0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9]

Input: [4, 4, 9, 7, 9, 7, 0, 0, 4, 9, 1, 7, 9, 5, 8, 7, 5, 6, 3, 8, 6, 7, 5, 8, 5, 0, 6, 3, 7, 0, 5, 3, 7, 5, 2, 4, 4, 9, 0, 7, 8, 2, 7, 7, 7, 2, 1, 3, 9, 9, 7, 9, 6, 6, 4, 5, 4, 2, 0, 8, 9, 0, 2, 2]
Incorrectly Sorted: [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
Reason: The incorrectly sorted list contains one extra 8 and is missing two 2s, one 3, three 4s, two 5s, one 6, six 7s and one 9.
Output: [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9]
</Examples>

Input: {input}
Incorrectly Sorted: {incorrectly_sorted}
"""

    got_split_prompt = """<Instruction> Split the following list of 32 numbers into 2 lists of 16 numbers each, the first list should contain the first 16 numbers and the second list the second 16 numbers.
Only output the final 4 lists in the following format without any additional text or thoughts!:
{{
    "List 1": [3, 4, 3, 5, 7, 8, 1, ...],
    "List 2": [2, 9, 2, 4, 7, 1, 5, ...]
}} </Instruction>

<Example>
Input: [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4, 5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
Output: 
{{
    "List 1": [9, 6, 7, 7, 2, 0, 2, 2, 3, 5, 0, 9, 2, 2, 4, 4],
    "List 2": [5, 2, 5, 1, 2, 8, 3, 8, 3, 9, 6, 0, 4, 2, 2, 3]
}}
</Example>

Input: {input}"""

    got_merge_prompt = """<Instruction> Merge the following 2 sorted lists of length {length1} each, into one sorted list of length {length2} using a merge sort style approach.
Only output the final merged list without any additional text or thoughts!:</Instruction>

<Approach>
To merge the two lists in a merge-sort style approach, foloow these steps:
1. Compare the first element of both lists.
2. Append the smaller element to the merged list and move to the next element in the list from which the smaller element came.
3. Repeat steps 1 and 2 until one of the lists is empty.
4. Append the remaining elements of the non-empty list to the merged list.
</Approach>

Merge the following two lists into one sorted list:
1: {input1}
2: {input2}

Merged list:
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If not exactly two thought states are provided.
        """
        assert len(state_dicts) == 2, "Expected two states for aggregation prompt."
        len_input1 = len(utils.string_to_list(state_dicts[0]["current"]))
        len_input2 = len(utils.string_to_list(state_dicts[1]["current"]))
        if len_input1 == len_input2:
            length = len_input1
        elif len_input1 + len_input2 - 32 <= 16:
            length = 16
        else:
            length = 32

        return self.got_merge_prompt.format(
            input1=state_dicts[0]["current"],
            input2=state_dicts[1]["current"],
            length1=length,
            length2=length * 2,
        )

    def generate_prompt(
        self, num_branches: int, original: str, current: str, method: str, **kwargs
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Input list of numbers.
        :type original: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """

        if current is None or current == "":
            input = original
        else:
            input = current
        if method.startswith("io"):
            return self.sort_prompt.format(input=input)
        elif method.startswith("cot"):
            return self.sort_prompt_cot.format(input=input)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.sort_prompt.format(input=input)
            return self.tot_improve_prompt.format(
                input=original,
                incorrectly_sorted=current,
                length=len(utils.string_to_list(original)),
            )
        elif method.startswith("got"):
            if current is None or current == "":
                return self.got_split_prompt.format(input=input)
            # if current is just a sublist of the original input, return the split prompt
            if kwargs["phase"] == 1:
                return self.sort_prompt.format(input=current)

            if (
                "unsorted_sublist" in kwargs
                and kwargs["unsorted_sublist"] != ""
                and len(kwargs["unsorted_sublist"]) < len(original) - 5
            ):
                original = kwargs["unsorted_sublist"]

            return self.tot_improve_prompt.format(
                input=original,
                incorrectly_sorted=current,
                length=len(utils.string_to_list(original)),
            )

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        """
        pass


class SortingParser(parser.Parser):
    """
    SortingParser provides the parsing of language model reponses specific to
    the sorting example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        :raise AssertionError: If not exactly two thought states are provided.
        """

        assert len(states) == 2, "Expected two states for aggregation answer."
        new_states = []
        for text in texts:
            answers = text.strip().split("\n")
            if any(["Output" in answer for answer in answers]):
                # cut elements until last output is found
                for answer in reversed(answers):
                    if "Output" in answer:
                        answers = answers[answers.index(answer) :]
                        break

            answers_stripped = [
                answer for answer in answers if "[" in answer and "]" in answer
            ]
            if len(answers_stripped) == 0:
                for answer in answers:
                    answer = "[" + answer + "]"
                    try:
                        answer_converted = utils.string_to_list(answer)
                        if len(answer_converted) > 0:
                            answers_stripped.append(answer)
                    except:
                        pass
            if len(answers_stripped) == 0:
                logging.warning(
                    f"Could not parse aggregation answer: {text}. Returning empty list."
                )
                answer = "[]"
            else:
                answer = [
                    answer[answer.index("[") : answer.index("]") + 1]
                    for answer in answers_stripped
                ][0]
            states = sorted(states, key=lambda x: x["part"])
            merged_unsorted_sublists = (
                states[0]["unsorted_sublist"][:-1]
                + ", "
                + states[1]["unsorted_sublist"][1:]
            )
            new_state = states[0].copy()
            new_state["current"] = answer
            new_state["unsorted_sublist"] = merged_unsorted_sublists
            new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            if state["method"] == "got" and state["current"] == "":
                # We expect a json which contains the four lists named "List 1" to "List 4"
                # cut everything until the opening bracket and everything after the closing bracket
                try:
                    text = text[text.index("{") : text.index("}") + 1]
                    json_dict = json.loads(text)
                    if len(json_dict.keys()) != 2:
                        logging.warning(
                            f"Expected 2 lists in json, but found {len(json_dict.keys())}."
                        )
                    for key, value in json_dict.items():
                        if "List" not in key:
                            logging.warning(
                                f"Expected key to contain 'List', but found {key}."
                            )
                            continue
                        if not isinstance(value, list):
                            value = utils.string_to_list(value)
                        new_state = state.copy()
                        new_state["current"] = str(value)
                        new_state["unsorted_sublist"] = str(value)
                        new_state["phase"] = 1
                        new_state["part"] = key
                        new_states.append(new_state)
                except Exception as e:
                    logging.error(
                        f"Could not parse step answer: {text}. Encountered exception: {e}"
                    )
            else:
                answers = text.strip().split("\n")
                answers = [
                    answer for answer in answers if "[" in answer and "]" in answer
                ]
                if any(["Output" in answer for answer in answers]):
                    # cut elements until last output is found
                    for answer in reversed(answers):
                        if "Output" in answer:
                            answers = answers[answers.index(answer) :]
                            break

                answers = [
                    answer[answer.index("[") : answer.index("]") + 1]
                    for answer in answers
                ]
                if len(answers) == 0:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning empty list."
                    )
                    answer = "[]"
                else:
                    if len(answers) > 1:
                        logging.warning(
                            f"Multiple answers found for step answer: {text}. Using the first one."
                        )
                    answer = answers[0]

                new_state = state.copy()
                new_state["current"] = answer
                new_state["phase"] = 2
                new_states.append(new_state)
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 20))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(1):
        operations_graph.append_operation(operations.Generate(1, 20))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT2 method.
    ToT2 uses a tree with more levels, but with fewer branches per level.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, 10))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    plans = operations.Generate(2, 1)
    operations_graph.append_operation(plans)  # generate the sublists
    for i in range(1, 3):
        list_id = f"List {i}"
        sub_list = operations.Selector(
            lambda thoughts, list_id=list_id: [
                thought for thought in thoughts if thought.state["part"] == list_id
            ]
        )
        sub_list.add_predecessor(plans)
        operations_graph.add_operation(sub_list)
        sort_sub_list = operations.Generate(1, 5)
        sort_sub_list.add_predecessor(sub_list)
        operations_graph.add_operation(sort_sub_list)
        score_sub_list = operations.Score(1, False, utils.num_errors)
        score_sub_list.add_predecessor(sort_sub_list)
        operations_graph.add_operation(score_sub_list)
        keep_best_sub_list = operations.KeepBestN(1, False)
        keep_best_sub_list.add_predecessor(score_sub_list)
        operations_graph.add_operation(keep_best_sub_list)

    final_aggregate = operations.Aggregate(10)
    operations_graph.append_operation(final_aggregate)
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_aggregate_final = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_aggregate_final)

    operations_graph.append_operation(operations.Generate(1, 10))
    score_aggr_3 = operations.Score(1, False, utils.num_errors)
    score_aggr_3.add_predecessor(keep_best_aggregate_final)
    operations_graph.append_operation(score_aggr_3)
    operations_graph.append_operation(operations.KeepBestN(1, False))

    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget
    path = os.path.join(os.path.dirname(__file__), "sorting_032.csv")
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), row[1], row[2]])

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    if not os.path.exists(os.path.join(os.path.dirname(__file__), "results")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "results"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"results/{extra_info}_{timestamp}"
    os.makedirs(os.path.join(os.path.dirname(__file__), folder_name))

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(
        os.path.join(os.path.dirname(__file__), folder_name, "config.json"), "w"
    ) as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=f"{folder_name}/log.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(
            os.path.join(os.path.dirname(__file__), folder_name, method.__name__)
        )

    for data in selected_data:
        logging.info(f"Running data {data[0]}: {data[1]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            lm = controller.ChatGPT(
                "../../graph_of_thoughts/controller/config.json",
                model_name=lm_name,
                cache=True,
            )
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                SortingPrompter(),
                SortingParser(),
                {
                    "original": data[1],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                os.path.dirname(__file__),
                folder_name,
                method.__name__,
                f"{data[0]}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    """
    Input (x)   : an unordered list of 32 numbers between 0 and 9 (inclusive)
    Output (y)  : a sorted list of 32 numbers between 0 and 9 (inclusive)
    Correct     : y == sorted(x)
    Input Example:
        [0, 1, 9, 4, 2, 2, 0, 5, 1...]
    Output Example:
        [0, 0, 0, 0, 1, 1, 1, 1, 2...]
    """
    budget = 30
    samples = [item for item in range(0, 100)]
    approaches = [io, cot, tot, tot2, got]

    spent = run(samples, approaches, budget, "chatgpt")

    logging.info(f"Spent {spent} out of {budget} budget.")