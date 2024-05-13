# Connect Melusine to a Gmail Mailbox

The Gmail connector has been developed to make it easier to get started with Melusine, by connecting directly to your mailbox.
The `GmailConnector` class allows:

- connect to your Gmail inbox
- retrieve the latest messages, filtering by date and labels
- extract relevant information from an e-mail, including attachments
- move messages individually into labels
- route an entire dataframe of e-mails to new labels, according to your detected use cases
- send emails

## Installation

First make sure to have a Gmail account and follow this [medium tutorial](https://medium.com/@preetipriyanka24/how-to-read-emails-from-gmail-using-gmail-api-in-python-20f7d9d09ae9) to get your `credentials.json` file or follow as below.

???note "Steps to create your credentials file from the medium post"

    - Sign in to [Google Cloud console](https://console.cloud.google.com/) and create a New Project or continue with an existing project.
    - Go to **APIs and Services**.
    - Go to **Enable APIs and Services**, enable Gmail API for the selected project.
    - Clicking on **OAuth Consent Screen** to configure the content screen.
    - Enter the Application name and save it.
    - Now go to **Credentials**.
    - Click on Create credentials, and go to **OAuth Client ID**.
        - Choose application type as Desktop Application.
        - Enter the Application name, and click on the Create button.
        - The Client ID will be created. Download it to your computer and save it as credentials.json
    - If the App is still in Testing mode, go to **OAuth Consent Screen** and add your gmail address to **Test users**.

Once your `credentials.json` created, save it to root for the first use.

## Usage

### First use

For the first use, a `token.json` will be created, save it. You will reuse it to sign in.

```Python
from melusine.connectors.gmail import GmailConnector
import logging

# Set up logging
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
connector_logger = logging.getLogger("melusine.connectors.gmail")
connector_logger.addHandler(ch)
connector_logger.setLevel(logging.INFO)

connector = GmailConnector(credentials_json_path="/Users/xxxxx/melusine/credentials.json")
# >>> 2024-05-06 11:18:58,636 - melusine.connectors.gmail - INFO - gmail token.json saved at: /Users/xxxxx/melusine
# >>> 2024-05-06 11:18:58,920 - melusine.connectors.gmail - INFO - Connected to mailbox: xxxxxxxxxx@gmail.com.

# Next usage will then be:
connector = GmailConnector(token_json_path="/Users/xxxxx/melusine/token.json")
```

!!! info
    A pop up window from google will ask you to choose your gmail account to sign in. If the app is still in testing mode, click on **Continue**.
    Then select all boxes to allow the credentials for read and modify rights, continue and close the window.

### Get emails

We have emails in the box to consider. These mails should either be put in **Melusine** label because they ask for something or in the **TRASH** label. Let's get the five last emails.

```Python
from melusine.connectors.gmail import GmailConnector
connector = GmailConnector(token_json_path="/Users/xxxxx/Desktop/melusine/token.json", done_label="TRASH")


df = connector.get_emails(max_emails=5)
# equivalent to: 
# df = connector.get_emails(max_emails=5, target_labels= ["INBOX"])
print(df)
```

|   | message_id | body                                       | header                | date                            | from                  | to                 | attachment |
|---|------------|--------------------------------------------|-----------------------|---------------------------------|-----------------------|--------------------|------------|
| 1 | 12456789   | This is a an example                       | Demo1                 | Mon, 13 May 2024 07:31:09 +0000 | <test@example.com>      | <mybox@melusine.com> | []         |
| 2 | 987654321  | I am very happy of this Melusine connector | Awesome connector!!   | Mon, 06 May 2024 10:55:22 +0000 | <connecting@people.com> | <mybox@melusine.com> | []         |
| 3 | 147258369  | Does Melusine is free ?                    | Impossible to believe | Thu, 02 May 2024 12:40:28 +0000 | <melusine_fan@maif.com> | <mybox@melusine.com> | []         |
| 4 | 741852963  | Hello World!                               | print                 | Mon, 29 Apr 2024 16:27:55 +0000 | <test@test.com>         | <mybox@melusine.com> | []         |
| 5 | 951753467  | Python is lovely                           | PEP                   | Thu, 25 Apr 2024 15:28:07 +0000 | <python@fan.com>        | <mybox@melusine.com> | []         |

And that's it, you have your last 5 mails from the INBOX!

!!! info "Filters"

    Filters can be used to get emails from specific labels. For example how about retrieveing emails unreaded in inbox from the last week?
    ```python
    df = connector.get_emails(
        max_emails=5,
        target_labels=["INBOX", "UNREAD"],
        start_date="2024/05/06",
        end_date="2024/05/12",
    )
    ```

!!! warning "Date format"
    When using `start_date` and `end_date` from `get_emails`, the format of the dates needs to be `YYYY/MM/DD` eg **2024/03/31**.

### Create label

To route emails to labels, they must exist. Using the example above, let's create the **Melusine** label.

```Python
connector.create_label("Melusine")
# >>> 2024-05-13 10:41:56,406 - melusine.connectors.gmail - INFO - Label Melusine has been created.
# >>> {'id': 'Label_4', 'name': 'Melusine', 'messageListVisibility': 'show', 'labelListVisibility': 'labelShow'}
```

The label has been created and it's id is `Label_4`.

### Route emails

There are two ways of routing emails, either individually using `move_to` or via the `route_emails` method, which takes as input a pandas data frame resulting from the application of the Melusine framework.

#### Route using Melusine output

Let's consider the same data frame, but with Melusine feedback to route each email via two detectors: **TRASH** and **Melusine**.
The column of interest is `target.` and `classified_df` is:

|   | message_id | body                                       | ... | attachment | target   |
|---|------------|--------------------------------------------|-----|------------|----------|
| 1 | 12456789   | This is a an example                       | ... | []         | TRASH    |
| 2 | 987654321  | I am very happy of this Melusine connector | ... | []         | Melusine |
| 3 | 147258369  | Does Melusine is free ?                    | ... | []         | Melusine |
| 4 | 741852963  | Hello World!                               | ... | []         | TRASH    |
| 5 | 951753467  | Python is lovely                           | ... | []         | TRASH    |

```Python
connector.route_emails(classified_emails=classified_df)
# >>> 2024-05-13 10:48:43,752 - melusine.connectors.gmail - INFO - Moved 3 emails to 'TRASH' label
# >>> 2024-05-13 10:48:44,110 - melusine.connectors.gmail - INFO - Moved 2 emails to 'Melusine' label
```

#### Route one by one (not recommanded)

Considering the above data frame, the first emails is not related to Melusine so let's move it to trash. Conversely, the second evokes Melusine and should have the **Melusine** label.

```Python
# First email
# as done_label is "TRASH"
connector.move_to_done(emails_id=[df.iloc[0].message_id])
# >>> 2024-05-13 10:48:58,870 - melusine.connectors.gmail - INFO - Moved 1 emails to 'TRASH' label

# Second email
connector.move_to(emails_id=[df.iloc[1].message_id], label_to_move_on="MELUSINE")
# >>> 2024-05-13 10:48:59,110 - melusine.connectors.gmail - INFO - Moved 1 emails to 'Melusine' label
```

!!! info
    You can route multiple emails since `emails_id` from `move_to` and `move_to_done` is a list.
