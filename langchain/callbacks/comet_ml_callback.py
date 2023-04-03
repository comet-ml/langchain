import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.utils import (
    BaseMetadataCallbackHandler,
    flatten_dict,
    hash_string,
    import_pandas,
    import_spacy,
    import_textstat,
    load_json,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult


def import_comet_ml() -> Any:
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use the comet_ml callback manager you need to have the `comet_ml` python "
            "package installed. Please install it with `pip install comet_ml`"
        )
    return comet_ml


class CometCallbackHandler(BaseMetadataCallbackHandler, BaseCallbackHandler):
    """Callback Handler that logs to Comet.

    Parameters:
        job_type (str): The type of comet_ml task such as "inference", "testing" or "qc"
        project_name (str): The comet_ml project name
        tags (list): Tags to add to the task
        task_name (str): Name of the comet_ml task
        visualize (bool): Whether to visualize the run.
        complexity_metrics (bool): Whether to log complexity metrics
        stream_logs (bool): Whether to stream callback actions to Comet

    This handler will utilize the associated callback method and formats
    the input of each callback function with metadata regarding the state of LLM run,
    and adds the response to the list of records for both the {method}_records and
    action. It then logs the response to Comet.
    """

    def __init__(
        self,
        task_type: Optional[str] = "inference",
        project_name: Optional[str] = "langchain_callback_demo",
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualize: bool = False,
        complexity_metrics: bool = False,
        stream_logs: bool = False,
    ) -> None:
        """Initialize callback handler."""

        comet_ml = import_comet_ml()
        spacy = import_spacy()
        super().__init__()

        self.task_type = task_type
        self.project_name = project_name
        self.tags = tags
        self.visualize = visualize
        self.complexity_metrics = complexity_metrics
        self.stream_logs = stream_logs

        self.temp_dir = tempfile.TemporaryDirectory()

        # Check if Comet task already exists (e.g. in pipeline)
        # TODO: DOUBLE CHECK
        if comet_ml.config.get_global_experiment():
            self.experiment = comet_ml.config.get_global_experiment()
        else:
            self.experiment = comet_ml.Experiment(  # type: ignore
                project_name=self.project_name,
            )
        if name:
            self.experiment.set_name(name)
        self.experiment.log_other("Created from", "langchain")
        warning = (
            "The comet_ml callback is currently in beta and is subject to change "
            "based on updates to `langchain`. Please report any issues to "
            "https://github.com/comet_ml/issue_tracking/issues with the tag `langchain`."
        )
        comet_ml.LOGGER.warning(warning)

        self.callback_columns: list = []
        self.action_records: list = []
        self.complexity_metrics = complexity_metrics
        self.visualize = visualize
        self.nlp = spacy.load("en_core_web_sm")

    def _init_resp(self) -> Dict:
        return {k: None for k in self.callback_columns}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        self.step += 1
        self.llm_starts += 1
        self.starts += 1

        metadata = self._init_resp()
        metadata.update({"action": "on_llm_start"})
        print("SERIALIZED", serialized)
        metadata.update(flatten_dict(serialized))
        metadata.update(self.get_custom_callback_meta())

        for prompt in prompts:
            prompt_resp = deepcopy(metadata)
            prompt_resp["prompts"] = prompt
            self.on_llm_start_records.append(prompt_resp)
            self.action_records.append(prompt_resp)
            # raise Exception(prompt_resp)
            # if self.stream_logs:
            #     self.experiment.log_text()
            #     self.logger.report_text(prompt_resp)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.step += 1
        self.llm_streams += 1

        resp = self._init_resp()
        resp.update({"action": "on_llm_new_token", "token": token})
        resp.update(self.get_custom_callback_meta())

        self.on_llm_token_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.step += 1
        self.llm_ends += 1
        self.ends += 1

        metadata = self._init_resp()
        metadata.update({"action": "on_llm_end"})
        metadata.update(flatten_dict(response.llm_output or {}))
        metadata.update(self.get_custom_callback_meta())

        for generations in response.generations:
            for generation in generations:
                # raise Exception(generation.dict())

                generation_resp = deepcopy(metadata)
                generation_resp.update(flatten_dict(generation.dict()))
                generation_resp.update(self.analyze_text(generation.text))
                self.on_llm_end_records.append(generation_resp)
                self.action_records.append(generation_resp)
                # if self.stream_logs:
                #     self.logger.report_text(generation_resp)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.step += 1
        self.errors += 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self.step += 1
        self.chain_starts += 1
        self.starts += 1

        resp = self._init_resp()
        resp.update({"action": "on_chain_start"})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())

        chain_input = inputs["input"]

        if isinstance(chain_input, str):
            input_resp = deepcopy(resp)
            input_resp["input"] = chain_input
            self.on_chain_start_records.append(input_resp)
            self.action_records.append(input_resp)
            if self.stream_logs:
                self.logger.report_text(input_resp)
        elif isinstance(chain_input, list):
            for inp in chain_input:
                input_resp = deepcopy(resp)
                input_resp.update(inp)
                self.on_chain_start_records.append(input_resp)
                self.action_records.append(input_resp)
                if self.stream_logs:
                    self.logger.report_text(input_resp)
        else:
            raise ValueError("Unexpected data format provided!")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self.step += 1
        self.chain_ends += 1
        self.ends += 1

        resp = self._init_resp()
        resp.update({"action": "on_chain_end", "outputs": outputs["output"]})
        resp.update(self.get_custom_callback_meta())

        self.on_chain_end_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        self.step += 1
        self.errors += 1

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        resp = self._init_resp()
        resp.update({"action": "on_tool_start", "input_str": input_str})
        resp.update(flatten_dict(serialized))
        resp.update(self.get_custom_callback_meta())

        self.on_tool_start_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.step += 1
        self.tool_ends += 1
        self.ends += 1

        resp = self._init_resp()
        resp.update({"action": "on_tool_end", "output": output})
        resp.update(self.get_custom_callback_meta())

        self.on_tool_end_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        self.step += 1
        self.errors += 1

    def on_text(self, text: str, **kwargs: Any) -> None:
        """
        Run when agent is ending.
        """
        self.step += 1
        self.text_ctr += 1

        resp = self._init_resp()
        resp.update({"action": "on_text", "text": text})
        resp.update(self.get_custom_callback_meta())

        self.on_text_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run when agent ends running."""
        self.step += 1
        self.agent_ends += 1
        self.ends += 1

        resp = self._init_resp()
        resp.update(
            {
                "action": "on_agent_finish",
                "output": finish.return_values["output"],
                "log": finish.log,
            }
        )
        resp.update(self.get_custom_callback_meta())

        self.on_agent_finish_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.step += 1
        self.tool_starts += 1
        self.starts += 1

        resp = self._init_resp()
        resp.update(
            {
                "action": "on_agent_action",
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
        )
        resp.update(self.get_custom_callback_meta())
        self.on_agent_action_records.append(resp)
        self.action_records.append(resp)
        if self.stream_logs:
            self.logger.report_text(resp)

    def analyze_text(self, text: str) -> dict:
        """Analyze text using textstat and spacy.

        Parameters:
            text (str): The text to analyze.

        Returns:
            (dict): A dictionary containing the complexity metrics.
        """
        resp = {}
        textstat = import_textstat()
        spacy = import_spacy()
        if self.complexity_metrics:
            text_complexity_metrics = {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "smog_index": textstat.smog_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "automated_readability_index": textstat.automated_readability_index(
                    text
                ),
                "dale_chall_readability_score": textstat.dale_chall_readability_score(
                    text
                ),
                "difficult_words": textstat.difficult_words(text),
                "linsear_write_formula": textstat.linsear_write_formula(text),
                "gunning_fog": textstat.gunning_fog(text),
                "text_standard": textstat.text_standard(text),
                "fernandez_huerta": textstat.fernandez_huerta(text),
                "szigriszt_pazos": textstat.szigriszt_pazos(text),
                "gutierrez_polini": textstat.gutierrez_polini(text),
                "crawford": textstat.crawford(text),
                "gulpease_index": textstat.gulpease_index(text),
                "osman": textstat.osman(text),
            }
            resp.update(text_complexity_metrics)

        if self.visualize and self.nlp and self.temp_dir.name is not None:
            doc = self.nlp(text)

            dep_out = spacy.displacy.render(  # type: ignore
                doc, style="dep", jupyter=False, page=True
            )
            dep_output_path = Path(
                self.temp_dir.name, hash_string(f"dep-{text}") + ".html"
            )
            dep_output_path.open("w", encoding="utf-8").write(dep_out)

            ent_out = spacy.displacy.render(  # type: ignore
                doc, style="ent", jupyter=False, page=True
            )
            ent_output_path = Path(
                self.temp_dir.name, hash_string(f"ent-{text}") + ".html"
            )
            ent_output_path.open("w", encoding="utf-8").write(ent_out)

            self.logger.report_media(
                "Dependencies Plot", text, local_path=dep_output_path
            )
            self.logger.report_media("Entities Plot", text, local_path=ent_output_path)

        return resp

    def _create_session_analysis_df(self) -> Any:
        """Create a dataframe with all the information from the session."""
        pd = import_pandas()
        on_llm_start_records_df = pd.DataFrame(self.on_llm_start_records)
        on_llm_end_records_df = pd.DataFrame(self.on_llm_end_records)

        llm_input_prompts_df = (
            on_llm_start_records_df[["step", "prompts", "name"]]
            .dropna(axis=1)
            .rename({"step": "prompt_step"}, axis=1)
        )
        complexity_metrics_columns = []
        visualizations_columns: List = []

        if self.complexity_metrics:
            complexity_metrics_columns = [
                "flesch_reading_ease",
                "flesch_kincaid_grade",
                "smog_index",
                "coleman_liau_index",
                "automated_readability_index",
                "dale_chall_readability_score",
                "difficult_words",
                "linsear_write_formula",
                "gunning_fog",
                "text_standard",
                "fernandez_huerta",
                "szigriszt_pazos",
                "gutierrez_polini",
                "crawford",
                "gulpease_index",
                "osman",
            ]

        llm_outputs_df = (
            on_llm_end_records_df[
                [
                    "step",
                    "text",
                    "token_usage_total_tokens",
                    "token_usage_prompt_tokens",
                    "token_usage_completion_tokens",
                ]
                + complexity_metrics_columns
                + visualizations_columns
            ]
            .dropna(axis=1)
            .rename({"step": "output_step", "text": "output"}, axis=1)
        )
        session_analysis_df = pd.concat([llm_input_prompts_df, llm_outputs_df], axis=1)
        # session_analysis_df["chat_html"] = session_analysis_df[
        #     ["prompts", "output"]
        # ].apply(
        #     lambda row: construct_html_from_prompt_and_generation(
        #         row["prompts"], row["output"]
        #     ),
        #     axis=1,
        # )
        return session_analysis_df

    def flush_tracker(
        self,
        langchain_asset: Any = None,
        name: Optional[str] = None,
        finish: bool = False,
    ) -> None:
        """Flush the tracker and setup the session.

        Everything after this will be a new table.

        Args:
            name: Name of the preformed session so far so it is identifyable
            langchain_asset: The langchain asset to save.
            finish: Whether to finish the run.

            Returns:
                None
        """
        pd = import_pandas()
        comet_ml = import_comet_ml()

        # Log the action records
        df = pd.DataFrame(self.action_records)
        self.experiment.log_table("langchain-action-record.json", df)
        # raise Exception(df)
        # self.logger.report_table(
        #     "Action Records", name, table_plot=pd.DataFrame(self.action_records)
        # )

        # Session analysis
        # session_analysis_df = self._create_session_analysis_df()
        # self.experiment.log_table("langchain-session-analysis.tsv", df)
        # self.logger.report_table(
        #     "Session Analysis", name, table_plot=session_analysis_df
        # )

        if self.stream_logs:
            self.logger.report_text(
                {
                    "action_records": pd.DataFrame(self.action_records),
                    "session_analysis": session_analysis_df,
                }
            )

        if langchain_asset:
            langchain_asset_path = Path(self.temp_dir.name, "model.json")
            try:
                langchain_asset.save(langchain_asset_path)
                # Create output model and connect it to the task
                # TODO: The Python SDK don't accept Pathlib.path objects?
                self.experiment.log_model(name, str(langchain_asset_path))
            except ValueError:
                langchain_asset.save_agent(langchain_asset_path)
                # TODO: The Python SDK don't accept Pathlib.path objects?
                self.experiment.log_model(name, str(langchain_asset_path))
            except NotImplementedError as e:
                print("Could not save model.")
                print(repr(e))
                pass

        # Cleanup after adding everything to Comet

        if finish:
            # Close old one
            self.experiment.end()

            # And recreate a new one
            self.experiment = comet_ml.Experiment(  # type: ignore
                    project_name=self.project_name,
                )
            if name:
                self.experiment.set_name(name)
        else:
            self.experiment.flush()

        # self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reset_callback_meta()