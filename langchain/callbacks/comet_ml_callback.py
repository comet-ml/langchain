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

COMPLEXITY_METRIC_NAMES = [
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
LANGCHAIN_MODEL_NAME = "langchain-model"


def import_comet_ml() -> Any:
    try:
        import comet_ml  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use the comet_ml callback manager you need to have the `comet_ml` python "
            "package installed. Please install it with `pip install comet_ml`"
        )
    return comet_ml


def _get_experiment(workspace=None, project_name=None):
    comet_ml = import_comet_ml()

    experiment = comet_ml.config.get_global_experiment()
    if experiment is None:
        experiment = comet_ml.Experiment(  # type: ignore
            workspace=workspace,
            project_name=project_name,
        )

    return experiment


def _fetch_text_complexity_metrics(text):
    textstat = import_textstat()
    text_complexity_metrics = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
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
    return text_complexity_metrics


def _summarize_metrics_for_generated_outputs(metrics):
    pd = import_pandas()
    metrics_df = pd.DataFrame(metrics)
    metrics_summary = metrics_df.describe()

    return metrics_summary.to_dict()


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
        workspace: Optional[str] = None,
        project_name: Optional[str] = "comet-langchain-demo",
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualizations: Optional[str] = None,
        complexity_metrics: bool = False,
        stream_logs: bool = False,
    ) -> None:
        """Initialize callback handler."""

        comet_ml = import_comet_ml()
        super().__init__()

        self.task_type = task_type
        self.workspace = workspace
        self.project_name = project_name
        self.tags = tags
        self.visualizations = visualizations
        self.complexity_metrics = complexity_metrics
        self.stream_logs = stream_logs
        self.temp_dir = tempfile.TemporaryDirectory()

        self.experiment = _get_experiment(workspace, project_name)
        self.experiment.log_other("Created from", "langchain")
        if tags:
            self.experiment.add_tags(tags)
        self.name = name
        if self.name:
            self.experiment.set_name(self.name)

        warning = (
            "The comet_ml callback is currently in beta and is subject to change "
            "based on updates to `langchain`. Please report any issues to "
            "https://github.com/comet_ml/issue_tracking/issues with the tag `langchain`."
        )
        comet_ml.LOGGER.warning(warning)

        self.callback_columns: list = []
        self.action_records: list = []
        self.complexity_metrics = complexity_metrics
        if self.visualizations:
            spacy = import_spacy()
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None

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

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        self.step += 1
        self.llm_streams += 1

        resp = self._init_resp()
        resp.update({"action": "on_llm_new_token", "token": token})
        resp.update(self.get_custom_callback_meta())

        self.action_records.append(resp)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.step += 1
        self.llm_ends += 1
        self.ends += 1

        metadata = self._init_resp()
        metadata.update({"action": "on_llm_end"})
        metadata.update(flatten_dict(response.llm_output or {}))
        metadata.update(self.get_custom_callback_meta())

        output_metrics = []
        for generations in response.generations:
            for generation in generations:
                text = generation.text

                generation_resp = deepcopy(metadata)
                generation_resp.update(flatten_dict(generation.dict()))

                text_metrics = self._get_text_metrics(text)
                generation_resp.update(text_metrics)
                self.action_records.append(generation_resp)
                self.on_llm_end_records.append(generation_resp)

                if text_metrics:
                    output_metrics.append(text_metrics)

        self._log_text_metrics(output_metrics, step=self.step)

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
            self.action_records.append(input_resp)

        elif isinstance(chain_input, list):
            for inp in chain_input:
                input_resp = deepcopy(resp)
                input_resp.update(inp)
                self.action_records.append(input_resp)

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

        self.action_records.append(resp)

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

        self.action_records.append(resp)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.step += 1
        self.tool_ends += 1
        self.ends += 1

        resp = self._init_resp()
        resp.update({"action": "on_tool_end", "output": output})
        resp.update(self.get_custom_callback_meta())

        self.action_records.append(resp)

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

        self.action_records.append(resp)

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
        self.action_records.append(resp)

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
        self.action_records.append(resp)

    def _get_text_metrics(self, text: str) -> dict:
        """Compute text metrics using textstat.

        Parameters:
            text (str): The text to analyze.

        Returns:
            (dict): A dictionary containing the complexity metrics.
        """
        resp = {}
        if self.complexity_metrics:
            text_complexity_metrics = _fetch_text_complexity_metrics(text)
            resp.update(text_complexity_metrics)

        return resp

    def flush_tracker(
        self,
        langchain_asset: Any = None,
        task_type: Optional[str] = "inference",
        workspace: Optional[str] = None,
        project_name: Optional[str] = "comet-langchain-demo",
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualizations: Optional[str] = None,
        complexity_metrics: bool = False,
        stream_logs: bool = False,
        finish: bool = False,
        reset: bool = False,
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
        self._log_session(langchain_asset)

        if langchain_asset:
            self._log_model(langchain_asset)

        if finish:
            self.experiment.end()

        if reset:
            self._reset(
                task_type,
                workspace,
                project_name,
                tags,
                name,
                visualizations,
                complexity_metrics,
                stream_logs,
            )

    def _log_stream(self, prompt_resp, step):
        self.experiment.log_text(prompt_resp, step=step)

    def _log_model(self, langchain_asset):
        comet_ml = import_comet_ml()

        self.experiment.log_parameters(langchain_asset.dict(), prefix="model")
        langchain_asset_path = Path(self.temp_dir.name, "model.json")
        model_name = self.name if self.name else LANGCHAIN_MODEL_NAME
        try:
            langchain_asset.save(langchain_asset_path)
            self.experiment.log_model(model_name, str(langchain_asset_path))

        except ValueError:
            langchain_asset.save_agent(langchain_asset_path)
            self.experiment.log_model(model_name, str(langchain_asset_path))

        except NotImplementedError as e:
            comet_ml.LOGGER.warning("Could not save Langchain Asset")

    def _log_session(self, langchain_asset):
        langchain_asset_parameters = langchain_asset.dict()
        num_generations_per_prompt = langchain_asset_parameters.get("n")

        dataframes_map = self._create_sessions_analysis_dataframe_map(
            num_generations_per_prompt
        )
        for key, dataframe in dataframes_map.items():
            self.experiment.log_table(f"langchain-{key}.json", dataframe)
            self.experiment.log_table(f"langchain-{key}.csv", dataframe)

        session_df = dataframes_map["llm_session"]
        self._log_visualizations(session_df)

    def _log_text_metrics(self, text_metrics, step):
        if not text_metrics:
            return

        text_metrics_summary = _summarize_metrics_for_generated_outputs(text_metrics)
        for key, value in text_metrics_summary.items():
            self.experiment.log_metrics(value, prefix=key, step=step)

    def _log_visualizations(self, session_df):
        if not (self.visualizations and self.nlp):
            return

        spacy = import_spacy()
        prompts = session_df["prompts"].tolist()
        outputs = session_df["text"].tolist()

        for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
            doc = self.nlp(output)
            sentence_spans = list(doc.sents)

            for visualization in self.visualizations:
                try:
                    html = spacy.displacy.render(
                        sentence_spans,
                        style=visualization,
                        options={"compact": True},
                        jupyter=False,
                        page=True,
                    )
                    self.experiment.log_asset_data(
                        html,
                        name=f"langchain-viz-{visualization}-{idx}.html",
                        metadata={"prompt": prompt},
                        step=idx,
                    )
                except Exception as e:
                    print(e)

        return

    def _reset(
        self,
        task_type: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        visualizations: Optional[str] = None,
        complexity_metrics: bool = False,
        stream_logs: bool = False,
    ):
        _task_type = task_type if task_type else self.task_type
        _workspace = workspace if workspace else self.workspace
        _project_name = project_name if project_name else self.project_name
        _tags = tags if tags else self.tags
        _name = name if name else self.name
        _visualizations = visualizations if visualizations else self.visualizations
        _complexity_metrics = (
            complexity_metrics if complexity_metrics else self.complexity_metrics
        )
        _stream_logs = stream_logs if stream_logs else self.stream_logs

        self.__init__(
            task_type=_task_type,
            workspace=_workspace,
            project_name=_project_name,
            tags=_tags,
            name=_name,
            visualizations=_visualizations,
            complexity_metrics=_complexity_metrics,
            stream_logs=_stream_logs,
        )

        self.reset_callback_meta()
        self.temp_dir = tempfile.TemporaryDirectory()

    def _create_sessions_analysis_dataframe_map(self, num_generations_per_prompt=1):
        pd = import_pandas()
        action_records_df = pd.DataFrame(self.action_records)

        llm_start_records_df = pd.DataFrame(self.on_llm_start_records)
        # Repeat each row based on the number of outputs generated per prompt
        # This is so the input prompt df aligns with the output df
        llm_start_records_df = llm_start_records_df.loc[
            llm_start_records_df.index.repeat(num_generations_per_prompt)
        ].reset_index(drop=True)
        llm_end_records_df = pd.DataFrame(self.on_llm_end_records)
        llm_session_df = pd.merge(
            llm_start_records_df,
            llm_end_records_df,
            left_index=True,
            right_index=True,
            suffixes=["_llm_start", "_llm_end"],
        )

        dataframe_map = dict(
            action_records=action_records_df, llm_session=llm_session_df
        )

        return dataframe_map
