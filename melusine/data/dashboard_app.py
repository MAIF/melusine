import ast
import inspect
import joblib
import pandas as pd

import textwrap

from collections import Counter
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder

from melusine.prepare_email.compute_complexity import (
    mean_words_by_sentence,
    structured_score,
)
from melusine.nlp_tools.tokenizer import Tokenizer

try:
    import streamlit as st
    from streamlit.logger import get_logger
except ModuleNotFoundError:
    raise (
        """Please install streamlit
        pip install melusine[viz]
        (or pip install streamlit)"""
    )

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    raise (
        """Please install plotly
        pip install melusine[viz]
        (or pip install plotly)"""
    )


tokenizer = Tokenizer(stop_removal=False)


def intro():
    st.sidebar.success("Select a dashboard above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a melusine dashboard from the dropdown on the left**
    """
    )


def exploration():
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Load Data
    status_text.text("Load Data")
    st.write("## Load Data 📥 ")
    st.write("Loading Data from :")
    data_path = st.text_input(
        "Pkl data path", "../../tutorial/data/emails_preprocessed.pkl"
    )
    df_emails_preprocessed = pd.read_pickle(data_path)
    progress_bar.progress(10)
    i = df_emails_preprocessed.shape[0]
    st.write("Dataset contains %i emails" % i)
    st.dataframe(df_emails_preprocessed.head())

    df_emails_preprocessed["structured_body"] = df_emails_preprocessed[
        "structured_body"
    ].apply(ast.literal_eval)
    progress_bar.progress(20)
    df_emails_preprocessed["mean_words_per_sentence"] = [
        mean_words_by_sentence(row, tokenizer)
        for index, row in df_emails_preprocessed.iterrows()
    ]
    df_emails_preprocessed["parts_tags"] = [
        structured_score(row) for index, row in df_emails_preprocessed.iterrows()
    ]
    df_emails_preprocessed[["parts_tags_set", "nb_parts_tags"]] = pd.DataFrame(
        df_emails_preprocessed["parts_tags"].tolist(),
        index=df_emails_preprocessed.index,
    )
    progress_bar.progress(30)

    # Build graphs
    st.write("## Build graphs 👩‍🎨")
    # Graphs counter
    i = 0

    # Sexe distribution
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Distribution of variable Sex")
    fig_sex = px.pie(df_emails_preprocessed, names="sexe")
    st.plotly_chart(fig_sex)

    # Age distribution
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Distribution of variable Age")
    fig_age = px.histogram(df_emails_preprocessed, x="age")
    st.plotly_chart(fig_age)

    # Analyse structuration of emails
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Structuration of emails")
    tags_counter = Counter()
    for email_tags in df_emails_preprocessed["parts_tags_set"]:
        tags_counter.update(email_tags)

    tags = list(tags_counter.keys())
    count = list(tags_counter.values())
    fig_parts = go.Figure([go.Bar(x=tags, y=count)])
    st.plotly_chart(fig_parts)

    # Analyse structuration of emails bis
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Number of parts tag by email")
    fig_nb_pt = px.histogram(df_emails_preprocessed, x="nb_parts_tags")
    st.plotly_chart(fig_nb_pt)

    # Analyse complexity of sentences
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Complexity of body sentences")
    fig_mw = px.histogram(df_emails_preprocessed, x="mean_words_per_sentence")
    st.plotly_chart(fig_mw)

    progress_bar.progress(100)
    status_text.text("Complete")
    st.button("Re-run")


def discrimination():
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Load Data
    status_text.text("Load Data")
    st.write("## Load Data 📥 ")
    st.write("Loading Data from :")
    data_path = st.text_input(
        "Data path", "../../tutorial/data/emails_preprocessed.csv"
    )
    df_emails_preprocessed = pd.read_csv(data_path, encoding="utf-8", sep=";")
    progress_bar.progress(10)
    i = df_emails_preprocessed.shape[0]
    st.write("Dataset contains %i emails" % i)
    st.dataframe(df_emails_preprocessed.head())

    df_emails_preprocessed["structured_body"] = df_emails_preprocessed[
        "structured_body"
    ].apply(ast.literal_eval)
    progress_bar.progress(20)
    df_emails_preprocessed["mean_words_per_sentence"] = [
        mean_words_by_sentence(row, tokenizer)
        for index, row in df_emails_preprocessed.iterrows()
    ]
    df_emails_preprocessed["parts_tags"] = [
        structured_score(row) for index, row in df_emails_preprocessed.iterrows()
    ]
    df_emails_preprocessed[["parts_tags_set", "nb_parts_tags"]] = pd.DataFrame(
        df_emails_preprocessed["parts_tags"].tolist(),
        index=df_emails_preprocessed.index,
    )
    progress_bar.progress(30)

    # The new clean_text column is the concatenation of the clean_header column and the clean_body column
    df_emails_preprocessed["clean_text"] = (
        df_emails_preprocessed["clean_header"]
        + " "
        + df_emails_preprocessed["clean_body"]
    )

    # Metadata input
    # By default the metadata used are :
    # - the extension : gmail, outlook, wanadoo..
    # - the day of the week at which the email has been sent
    # - the hour at which the email has been sent
    # - the minute at which the email has been sent
    status_text.text("Load Metadata")
    st.write("## Load Metadata 📥 ")
    st.write("Loading Metadata from :")
    metadata_path = st.text_input("Metadata path", "../../tutorial/data/metadata.csv")
    df_meta = pd.read_csv(metadata_path, encoding="utf-8", sep=";")

    st.write(
        "X is a Pandas dataframe with a clean_text column that will be used for the text input and columns \
    containing the dummified metadata"
    )
    X = pd.concat([df_emails_preprocessed["clean_text"], df_meta], axis=1)
    st.dataframe(X.head())

    st.write("y is a numpy array containing the encoded labels")
    y = df_emails_preprocessed["label"]
    st.dataframe(y)

    # Loading the neural network
    status_text.text("Load Model")
    st.write("## Load Neural Network 🔮 ")
    st.write("Loading model pickel from :")
    nn_model_path = st.text_input("Model path", "../../tutorial/data/nn_model.pickle")
    # The NeuralModel saved as a pickle file has to be loaded first
    nn_model = joblib.load(nn_model_path)
    # Then the Keras model and its weights can be loaded
    st.write("Loading weights  from :")
    weights_path = st.text_input("Weights path", "../../tutorial/data/nn_model")
    nn_model.load_nn_model(weights_path)

    # Generate Label Encoder
    le = LabelEncoder()
    le.fit_transform(y)

    y_res = nn_model.predict(X)
    y_res = le.inverse_transform(y_res)
    df_emails_preprocessed["prediction"] = y_res
    df_emails_preprocessed["prediction_error"] = y_res != y
    st.write("Model prediction over the choosen dataset")
    st.dataframe(
        df_emails_preprocessed[
            ["clean_text", "label", "prediction", "prediction_error"]
        ]
    )

    # Build graphs
    st.write("## Build graphs 👩‍🎨")
    # Graphs counter
    i = 0

    # Analyse Prediction Error
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Prediction Error rate ")
    label_counter = Counter()
    error_counter = Counter()
    for label, error in zip(
        df_emails_preprocessed["label"],
        df_emails_preprocessed["prediction_error"],
    ):
        label_counter.update([label])
        if error == 1:
            error_counter.update([label])
    labels = list(label_counter.keys())
    count = list(label_counter.values())
    labels_error = list(error_counter.keys())
    count_error = list(error_counter.values())
    fig_label_error = go.Figure(
        data=[
            go.Bar(x=labels, y=count, name="label count"),
            go.Bar(x=labels_error, y=count_error, name="prediction error"),
        ]
    )
    fig_label_error.update_layout(barmode="overlay")
    fig_label_error.update_traces(opacity=0.75)
    st.plotly_chart(fig_label_error)

    # Sexe distribution
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Error rate in regards of variable Sex")
    error_counter = Counter()
    sex_counter = Counter()
    for sex, error in zip(
        df_emails_preprocessed["sexe"],
        df_emails_preprocessed["prediction_error"],
    ):
        sex_counter.update([sex])
        if error == 1:
            error_counter.update([sex])
    sex = list(sex_counter.keys())
    count = list(sex_counter.values())
    sex_error = list(error_counter.keys())
    count_error = list(error_counter.values())
    fig_sex = go.Figure(
        data=[
            go.Bar(x=sex, y=count, name="sex count"),
            go.Bar(x=sex_error, y=count_error, name="prediction error"),
        ]
    )
    fig_sex.update_layout(barmode="overlay")
    fig_sex.update_traces(opacity=0.75)
    st.plotly_chart(fig_sex)

    # Age distribution
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Error rate in regards of variable Age")
    fig_age = px.histogram(df_emails_preprocessed, x="age", color="prediction_error")
    fig_age.update_layout(barmode="group")
    st.plotly_chart(fig_age)

    # Analyse structuration of emails bis
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Error rate in regards of number of parts tag by email")
    fig_nb_pt = px.histogram(
        df_emails_preprocessed, x="nb_parts_tags", color="prediction_error"
    )
    fig_nb_pt.update_layout(barmode="group")
    st.plotly_chart(fig_nb_pt)

    # Analyse complexity of sentences
    i += 1
    status_text.text("Build graphs %i" % i)
    st.write("### Error rate in regards of the complexity of body sentences")
    fig_mw = px.histogram(
        df_emails_preprocessed,
        x="mean_words_per_sentence",
        color="prediction_error",
    )
    fig_mw.update_layout(barmode="group")
    st.plotly_chart(fig_mw)

    progress_bar.progress(100)
    status_text.text("Complete")
    st.button("Re-run")


LOGGER = get_logger(__name__)

# Dictionary of dashboard
DASHBOARDS = OrderedDict(
    [
        ("—", (intro, None)),
        (
            "Exploration dahsboard",
            (
                exploration,
                """
Important graphs on email structuring and complexity with some simple graphics on emails dataset
""",
            ),
        ),
        (
            "Discrimination dashboard",
            (
                discrimination,
                """
Graphs studying the discrimination relationships of the previous variables and the classification error on emails dataset
""",
            ),
        ),
    ]
)


def run():
    db_name = st.sidebar.selectbox("Choose a dashboard", list(DASHBOARDS.keys()), 0)
    dashboard = DASHBOARDS[db_name][0]

    if db_name == "—":
        show_code = False
        st.write("# Welcome to Melusine Streamlit App! 👋")
    else:
        show_code = st.sidebar.checkbox("Show code", False)
        st.markdown("# %s" % db_name)
        description = DASHBOARDS[db_name][1]
        if description:
            st.write(description)
        # Clear everything from the intro page.
        # We only have 4 elements in the page so this is intentional overkill.
        for i in range(10):
            st.empty()

    dashboard()

    if show_code:
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(dashboard)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


if __name__ == "__main__":
    run()
