=====
Usage
=====

To use melusine in a project::

    import melusine

Melusine input data : Email DataFrames
--------------------------------------

The basic requirement to use Melusine is to have an input e-mail DataFrame with the following columns:

    - ``body``  : Body of an email (single message or conversation historic)
    - ``header``: Header of an email
    - ``date``  : Reception date of an email
    - ``from``  : Email address of the sender
    - ``to``    : Email address of the recipient
    - ``label`` (optional) : Label of the email for a classification task (examples: Business, Spam, Finance or Family)

.. csv-table::
    :header: body, header, date, from, to, label

    "Thank you.\\nBye,\\nJohn", "Re: Your order", "jeudi 24 mai 2018 11 h 49 CEST", "anonymous.sender@unknown.com", "anonymous.recipient@unknown.fr", "A"

In the examples presented below, a toy email DataFrame containing anonymized emails is used.
The toy DataFrame can be loaded as follows::

    import melusine
    import pandas as pd

    df_email = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')
    df_email.head()


Prepare email subpackage : Basic usage
--------------------------------------

A common pre-processing step is to check whether an e-mail is an answer or not.
This can be done in Melusine with the function :ref:`add_boolean_answer<manage_transfer_reply>`::

    from melusine.prepare_email.manage_transfer_reply import add_boolean_answer

    df_email['is_answer'] = df_email.apply(add_boolean_answer, axis=1)


A new column ``is_answer`` is created containing a boolean variable:

    - True if the message is an answer
    - False if the message is not an answer

.. csv-table::
    :header: body, header, is_answer

    "Happy Birthday Bill!!", "Birthday", False
    "Thank you", "Re: Birthday", True

Create an email pre-processing pipeline
---------------------------------------

An email pre-processing pipeline takes an email DataFrame as input and executes a sequence of *Transformers*
on every email in the DataFrame.
The recommended way to create a pre-processing pipeline with Melusine is to:

    1. Wrap pre-processing functions in :ref:`TransformerScheduler<transformerScheduler>` objects.
    2. Use a `Scikit-Learn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ object to chain transformers

Once the pipeline has been set-up, the pre-processing of an email DataFrame is straightforward:

    >>> df_email_preprocessed = pipeline.fit_transform(df_email)

TransformerScheduler class
^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions can be wrapped in a :ref:`TransformerScheduler<transformerScheduler>` object that can be integrated into an execution Pipeline.
``TransformerScheduler`` objects are compatible with the `scikit-learn API <https://scikit-learn.org/stable/>`_
(they have fit and transform methods).


A ``TransformerScheduler`` object is initialized with a functions_scheduler argument.
The functions_scheduler argument is a list of tuples containing information about the desired pre-processing functions.
Each tuple describe an individual function and should contain the following elements:

    1. A function
    2. A tuple with the function's arguments
       (if no arguments are required, use None or an empty tuple)
    3. A column(s) name list returned by the function
       (if no arguments are required, use None or an empty list)

The code below describes the definition of a transformer::

    from melusine.utils.transformer_scheduler import TransformerScheduler

    melusine_transformer = TransformerScheduler(
    functions_scheduler=[
        (my_function_1, (argument1, argument2), ['return_col_A']),
        (my_function_2, None, ['return_col_B', 'return_col_C'])
        (my_function_3, (argument1, ), None),
    mode='apply_by_multiprocessing',
    n_jobs=4)
    ])

The other parameters of the *TransformerScheduler* class are:

    - ``mode`` (optional): Define mode to apply function along a row axis (axis=1)
      If set to 'apply_by_multiprocessing', it uses multiprocessing tool to parallelize computation.
      Possible values are 'apply' (default) and 'apply_by_multiprocessing'

    - ``n_jobs`` (optional): Number of cores used for computation. Default value, 1.
      Possible values are integers ranging from 1 (default) to the number of cores available for computation

A TransformerScheduler can be used independently or included in a scikit pipeline (recommended):

    >>> # Used independently
    >>> df_email = melusine_transformer.fit_transform(df_email)

    >>> # Used in a scikit pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([('MelusineTransformer', melusine_transformer)])
    >>> df_email = pipeline.fit_transform(df_email)

The *fit_transform* method returns a DataFrame with new features (new columns)

.. csv-table::
    :header: body, header, return_col_A, return_col_B, return_col_C, return_col_D

    "Happy Birthday Bill!!", "Birthday", "new_feature_A", "new_feature_B", "new_feature_C", "new_feature_D"
    "Thank you", "Re: Birthday", "new_feature_A", "new_feature_B", "new_feature_C", "new_feature_D"


Chaining transformers in a scikit-learn pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once all the desired functions and transformers have been defined, transformers can be chained in a `Scikit-Learn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.
The code below describes the definition of a pipeline::

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
    ('TransformerName1', TransformerObject1),
    ('TransformerName2', TransformerObject2),
    ('TransformerName3', TransformerObject3),
    ])

Example of a working pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A working pre-processing pipeline is given below::

    from sklearn.pipeline import Pipeline
    from melusine.utils.transformer_scheduler import TransformerScheduler
    from melusine.prepare_email.manage_transfer_reply import add_boolean_answer, add_boolean_transfer
    from melusine.prepare_email.build_historic import build_historic
    from melusine.prepare_email.mail_segmenting import structure_email

    ManageTransferReply = TransformerScheduler(
    functions_scheduler=[
        (add_boolean_answer, None, ['is_answer']),
        (add_boolean_transfer, None, ['is_transfer'])
    ])

    HistoricBuilder = TransformerScheduler(
    functions_scheduler=[
        (build_historic, None, ['structured_historic']),
    ])

    Segmenting = TransformerScheduler(
    functions_scheduler=[
        (structure_email, None, ['structured_body'])
    ])

    prepare_data_pipeline = Pipeline([
    ('ManageTransferReply', ManageTransferReply),
    ('HistoricBuilder', HistoricBuilder),
    ('Segmenting', Segmenting),
    ])

    df_email = prepare_data_pipeline.fit_transform(df_email)

In this example, the pre-processing functions applied are:

    - :ref:`add_boolean_answer<manage_transfer_reply>` : Email is an answer (True/False)
    - :ref:`add_boolean_transfer<manage_transfer_reply>` : Email is transferred (True/False)
    - :ref:`build_historic<build_historic>` : When email is a conversation, reconstructs the individual message historic
    - :ref:`structure_email<mail_segmenting>` : Splits parts of each messages in historic and tags them (tags: Hello, Body, Greetings, etc)

Create a custom email pre-processing function
----------------------------------------------

Creating a custom pre-processing function and adding it to a pre-processing pipeline can be done easily with *Melusine*.
Two main requirements are:

    1. Make the function compatible with the pandas apply method
        * First argument should be 'row' (Row of an email DataFrame)
            >>> def my_function(row, arg1, arg2):
        * Example: row['header'] will contain the header of a message
    2. Make sure to call existing columns of the DataFrame
        * Don't call row['is_answer'] before the 'is_answer' column has been created

The following example creates a custom function to count the occurrence of a word in the body of an email::

    from sklearn.pipeline import Pipeline
    from melusine.utils.transformer_scheduler import TransformerScheduler
    from melusine.prepare_email.manage_transfer_reply import add_boolean_answer, add_boolean_transfer

    # Create a fake email Dataframe
    df_duck = pd.DataFrame({
        "body" : ["Lion Duck Pony", "Duck Pony Pony", "Duck Duck Pony"],
        "header" : ["zoo report", "Tr : zoo report", "Re : zoo report"]
    })

    # Define a custom function
    def count_word_occurrence_in_body(row, word):
        all_word_list = row["body"].lower().split()
        word_occurence = all_word_list.count(word)
        return word_occurence

    # Wrap function in a transformer
    CountWordOccurrence = TransformerScheduler(
    functions_scheduler=[
        (count_word_occurrence_in_body, ("duck",), ['duck_count']),
        (count_word_occurrence_in_body, ("pony",), ['pony_count']),
    ])

    # Create a second transformer with regular Melusine functions
    ManageTransferReply = TransformerScheduler(
    functions_scheduler=[
        (add_boolean_answer, None, ['is_answer']),
        (add_boolean_transfer, None, ['is_transfer'])
    ])

    # Chain transformers in a pipeline
    prepare_data_pipeline = Pipeline([
        ('CountWordOccurrence', CountWordOccurrence), # Transformer with custom functions
        ('ManageTransferReply', ManageTransferReply), # Transformer with regular Melusine functions
    ])

    # Pre-process input DataFrame
    df_duck_prep = prepare_data_pipeline.fit_transform(df_duck)

.. csv-table::
    :header: body, header, duck_count, pony_count, is_answer, is_transfer

    "Lion Duck Pony", "zoo report", "1", "1", False, False
    "Duck Duck Pony", "Re : zoo report", "2", "1", "True", "False"
    "Duck Pony Pony", "Tr : zoo report", "1", "2", False, False

Note : It is totally fine to mix regular and custom functions in a transformer.

Testing a function on a single email
------------------------------------

Since all pre-processing functions are made compatible with pandas apply function,
a function can be tested on a single email.
In the example below, the function :ref:`add_boolean_answer<manage_transfer_reply>` is tested on a single email::

    from melusine.prepare_email.manage_transfer_reply import add_boolean_answer

    email_index = 2
    email_is_answer = add_boolean_answer(df_email.iloc[email_index])
    print("Message %d is an answer: %r" %(email_index, email_is_answer))

Output::

    "Message 2 is an answer: True"



NLP tools subpackage
--------------------

The different classes of the NLP tools subpackage are described in this section.

Phraser
^^^^^^^

The Melusine :ref:`Phraser <phraser>` class transforms common multi-word expressions into single elements:

    >>> new york -> new_york

To train a Melusine Phraser (which is based on a `Gensim Phraser <https://www.pydoc.io/pypi/gensim-3.2.0/autoapi/models/phrases/index.html>`_),
the input email DataFrame should contain a 'clean_body' column which can be created with the :ref:`clean_body<cleaning>` function.

In the example below, a Phraser is trained on a toy DataFrame::

    from melusine.nlp_tools.phraser import Phraser
    from melusine.nlp_tools.phraser import phraser_on_text

    phraser = Phraser()
    df_new_york = pd.DataFrame({
        'clean_body' : ["new york is so cool", "i love new york", "new york city"]
    })

    phraser.train(df_new_york)

    df_new_york['clean_body'] = df_new_york['clean_body'].apply(phraser_on_text, args=(phraser,))

    # Save the Phraser instance to disk
    phraser.save(filepath)
    # Load the Phraser
    phraser = phraser().load(filepath)

In reality, a training set with only 3 emails is too small to train a Phraser.
For illustrative purpose, the table below shows the expected output.

.. csv-table::
    :header: clean_body, clean_body_new

    "new york is so cool", "new_york is so cool"
    "i love new york",  "i love new_york"
    "new york city", "new_york city"

The specific parameters of the :ref:`Phraser <phraser>` class are:

    - *common_terms* : list of stopwords to be ignored (default value = stopword list from NLTK)
    - *threshold* : threshold to select collocations
    - *min_count* : minimum count of word to be selected as collocation

Tokenizer
^^^^^^^^^

A tokenizer splits a sentence-like string into a list of sub-strings (tokens).
The Melusine :ref:`Tokenizer <tokenizer>` class is based on a `NLTK regular expression tokenizer <https://www.nltk.org/api/nltk.tokenize.html>`_
which uses a regular expression (regex) pattern to tokenize the text::

    from melusine.nlp_tools.tokenizer import Tokenizer

    df_tok = pd.DataFrame({
        'clean_body' : ["hello, i'm here to tokenize text. bye"],
        'clean_header' : ["re: hello"],
    })

    tokenizer = Tokenizer(columns=['clean_body', 'clean_header'])
    df_tok = tokenizer.fit_transform(df_tok)

A new column ``tokens`` is created with a list of the tokens extracted from the text data.

.. csv-table::
    :header: clean_body, clean_header, tokens

    "hello, i'm here to tokenize text. bye", "re: hello", "['re', 'hello', 'hello', 'i', 'm', 'here', 'to', 'tokenize', 'text', 'bye']"

The specific parameters of the :ref:`Tokenizer <tokenizer>` class are:

    - *stopwords* : list of keywords to be ignored (this list can be defined in the conf file)
    - *stop_removal* : True if stopwords should be removed, else False

Embeddings
^^^^^^^^^^

With a regular representation of words, there is one dimension for each word in the vocabulary
(set of all the words in a text corpus).
The computational cost of NLP tasks, such as training a neural network model, based on such a high dimensional space can be prohibitive.
`Word embeddings <https://en.wikipedia.org/wiki/Word_embedding>`_ are abstract representations of words in a lower dimensional vector space.
One of the advantages of word embeddings is thus to save computational cost.

The Melusine :ref:`Embedding <embedding>` class uses the `Gensim Word2Vec module <https://radimrehurek.com/gensim/models/word2vec.html>`_ to train a `word2vec model <https://en.wikipedia.org/wiki/Word2vec>`_.
The trained Embedding object will be used in the :ref:`Models<models>` subpackage to train a Neural Network to classify emails.

The code below illustrates how the Embeddings class works. It should be noted that, in practice, to train a word embedding model, a lot of emails are required::

    from melusine.nlp_tools.embedding import Embedding

    df_embeddings = pd.DataFrame({
        'clean_body' : ["word text word text data word text"],
        'clean_header' : ["re: hello"],
    })

    embedding = Embedding(columns='clean_body', min_count=3)
    embedding = embedding.train(df_embeddings)

    # Save the trained Embedding instance to disk
    embedding.save('filepath')

    # Load the trained Embedding instance
    embedding = Embedding().load(filepath)

    # Use trained Embedding to initialise the Neural Network Model
    # The definition of a neural network model is not discussed in this section
    nn_model : NeuralModel("...", pretrained_embedding=embedding, "...")

Summarizer subpackage
---------------------

The main item of the :ref:`Summarizer<summarizer>` subpackage is the :ref:`KeywordGenerator<keywords_generator>` class.
The KeywordGenerator class extracts relevant keywords in the text data based on a `tf-idf <https://en.wikipedia.org/wiki/Tf–idf>`_ score.

Requirements on the input DataFrame to use a KeywordGenerator:

    - KeywordGenerator requires a 'tokens' column which can be generated with a :ref:`Tokenizer <tokenizer>`

Keywords can then be extracted as follows::

    from melusine.summarizer.keywords_generator import KeywordsGenerator
    from melusine.nlp_tools.tokenizer import Tokenizer


    df_zoo = pd.DataFrame({
        'clean_body' : ["i like lions and ponys and gorillas", "i like ponys and tigers", "i like tigers and lions", "i like raccoons and unicorns"],
        'clean_header' : ["things i like", "things i like", "things i like", "things i like"]
    })

    tokenizer = Tokenizer(columns=['clean_body', 'clean_header'])
    # Create the 'tokens' column
    df_zoo = tokenizer.fit_transform(df_zoo)

    keywords_generator = KeywordsGenerator(n_max_keywords=2, stopwords=['like'])
    # Fit keyword generator on the text data corpus (using the tokens column)
    keywords_generator.fit(X)
    # Extract relevant keywords
    keywords_generator.transform(X)

In the text data of the example, some words are very common such as "i", "like" or "things", whereas other words are rare, such as "raccoons".
The keyword generator prioritise the rare words in the keyword extraction process:

.. csv-table::
    :header: clean_body, clean_header, tokens, keywords

    "i like lions and ponies and gorillas", "things i like", "[things, i, i, lions, and, ponies, and, gorillas]", "[lions, ponys]"
    "i like ponies and tigers", "things i like", "[things, i, i, ponies, and, tigers]", "[ponies, tigers]"
    "i like tigers and lions", "things i like", "[things, i, i, tigers, and, lions]", "[tigers, lions]"
    "i like raccoons and unicorns", "things i like", "[things, i, i, raccoons, and, unicorns]", "[raccoons, unicorns]"

The specific parameters of the :ref:`KeywordGenerator<keywords_generator>` class are:

    - *max_tfidf_features* : size of vocabulary for tfidf
    - *keywords* : list of keyword to be extracted in priority (this list can be defined in the conf file)
    - *stopwords* : list of keywords to be ignored (this list can be defined in the conf file)
    - *resample* : when DataFrame contains a 'label' column, balance the dataset by resampling
    - *n_max_keywords* : maximum number of keywords to be returned for each email
    - *n_min_keywords* : minimum number of keywords to be returned for each email
    - *threshold_keywords* : minimum tf-idf score for a word to be selected as keyword

Models subpackage
-----------------

The main item of the Models subpackage is the :ref:`NeuralModel <train>` class.
The NeuralModel creates a Neural Network that can be trained and used to classify emails.

The minimum input features required by the NeuralModel class are the following:

    - An email DataFrame with:

        - an integer 'label' column (a label encoder can be used to convert class names into integers)
        - a 'clean_text' column with text data
    - An instance of the :ref:`Embedding <embedding>` class (Trained word embedding model)

The code below shows a minimal working example for Email Classification using a NeuralModel instance (a much larger training set is required to obtain meaningful results)::


    # Prepare email
    from melusine.utils.transformer_scheduler import TransformerScheduler
    from melusine.prepare_email.manage_transfer_reply import \
        check_mail_begin_by_transfer, update_info_for_transfer_mail, add_boolean_answer, add_boolean_transfer
    from melusine.prepare_email.build_historic import build_historic
    from melusine.prepare_email.mail_segmenting import structure_email
    from melusine.prepare_email.body_header_extraction import extract_last_body, extract_header
    from melusine.prepare_email.cleaning import clean_body, clean_header
    from melusine.prepare_email.metadata_engineering import MetaDate, MetaExtension, Dummifier

    # Scikit-Learn API
    from sklearn.pipeline import Pipeline

    # NLP tools
    from melusine.nlp_tools.tokenizer import Tokenizer
    from melusine.nlp_tools.embedding import Embedding

    # Summarizer
    from melusine.summarizer.keywords_generator import KeywordsGenerator

    # Models
    from melusine.models.train import NeuralModel
    from melusine.models.neural_architectures import cnn_model

    X = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')

    # Convert 'label' column to integer values
    X['label'] = X_train['label'].astype("category").cat.codes

    # Prepare mail
    ManageTransferReply = TransformerScheduler(
        functions_scheduler=[
            (check_mail_begin_by_transfer, (), ['is_begin_by_transfered']),
            (update_info_for_transfer_mail, (), None),
            (add_boolean_answer, (), ['is_answer']),
            (add_boolean_transfer, (), ['is_transfer'])
        ],
        mode='apply_by_multiprocessing',
        n_jobs=4)

    HistoricBuilder = TransformerScheduler(
        functions_scheduler=[
            (build_historic, (), ['structured_historic']),
        ],
        mode='apply_by_multiprocessing',
        n_jobs=4)

    Segmenting = TransformerScheduler(
        functions_scheduler=[
            (structure_email, (), ['structured_body'])
        ],
        mode='apply_by_multiprocessing',
        n_jobs=4)

    GetLastBodyHeader = TransformerScheduler(
        functions_scheduler=[
            (extract_last_body, (), ['last_body']),
            (extract_header, (), ['last_header'])
        ],
        mode='apply_by_multiprocessing',
        n_jobs=4)

    Cleaner = TransformerScheduler(
        functions_scheduler=[
            (clean_body, (), ['clean_body']),
            (clean_header, (), ['clean_header']),
        ],
        mode='apply_by_multiprocessing',
        n_jobs=4)

    prepare_data_pipeline = Pipeline([
        ('ManageTransferReply', ManageTransferReply),
        ('HistoricBuilder', HistoricBuilder),
        ('Segmenting', Segmenting),
        ('GetLastBodyHeader', GetLastBodyHeader),
        ('Cleaner', Cleaner),
        ('MetaExtension', MetaExtension()),
        ('MetaDate', MetaDate()),
    ])

    # Run prepare email pipeline
    X = prepare_data_pipeline.fit_transform(X)

    # Dummify categorical data
    categorical_cols = [cols for cols in ['extension', 'dayofweek', 'hour'] if cols in X.columns]
    X_dummy = Dummifier(columns_to_dummify=categorical_cols).fit_transform(X)

    # Concatenate dummified features with original features
    X_train = pd.concat([X, X_dummy], axis=1)

    # Create and train a word embedding model
    embedding = Embedding(columns='clean_body', min_count=2)
    embedding = embedding.train(X_train)

    def concatenate_body_header(row):
    """Concatenate header content and body content."""
    clean_text = row['clean_header'] + " // " + row['clean_body']
    return clean_text

    X_train['clean_text'] = X_train.apply(concatenate_body_header, axis=1)

    # List of columns containing meta-data
    list_meta = ['extension', 'dayofweek', 'hour']

    # Instanciate a NeuralModel instance with a CNN (imported from the neural_architectures module), an embedding and a list od meta data as arguments
    nn_model = NeuralModel(cnn_model, embedding, list_meta = list_meta)

    # Train the NeuralModel
    nn_model.fit(X_train.drop(['label'], axis=1), X_train['label'])

    # Make a prediction with the trained model
    y_res = nn_model.predict(X_train.drop(['label'], axis=1))


TODO : Describe NeuralModel parameters
