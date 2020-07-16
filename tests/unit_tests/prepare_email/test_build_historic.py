import pandas as pd
from melusine.prepare_email.build_historic import build_historic

body = "En date de lun. 10 mai 2018 LeSociétaire a écrit\nObjet : Demande de régularisation \n  \n  \n Bonjours, \n  \n Suite a notre conversation téléphonique \
de Mardi , pourriez vous me dire la \n somme que je vous dois afin d'd'être \
en régularisation . \n  \n Merci bonne journée \n  \n Le mar. 22 mai 2018 \
à 10:20,  <gestionsocietaire@mutuelle.fr> a écrit\xa0: \n\nObjet : Envoi d'un document \n\n Bonjour. \n  \n \
Merci de bien vouloir prendre connaissance du document ci-joint : \n 1 - \
Relevé d'identité postal MUTUELLE (contrats) \n  \n Sentiments \
mutualistes. \n  \n La Mututelle \n  \n La visualisation des fichiers PDF \
nécessite Adobe Reader. \n  "


output = [
    {
        "text": "Bonjours, \n  \n Suite a notre conversation \
téléphonique de Mardi , pourriez vous me dire la \n somme que je vous \
dois afin d'd'être en régularisation . \n  \n Merci bonne journée",
        "meta": "En date de lun. 10 mai 2018 LeSociétaire a écrit\nObjet : Demande de régularisation \n  \n  \n ",
    },
    {
        "text": "Bonjour. \n  \n Merci de bien vouloir prendre connaissance \
du document ci-joint : \n 1 - Relevé d'identité postal MUTUELLE \
(contrats) \n  \n Sentiments mutualistes. \n  \n La Mututelle \n  \n \
La visualisation des fichiers PDF nécessite Adobe Reader. \n  ",
        "meta": " \n  \n Le mar. 22 mai 2018 à 10:20,  \
<gestionsocietaire@mutuelle.fr> a écrit\xa0: \n\nObjet : Envoi d'un document \n\n ",
    },
]


empty_body = "\n\nA :  societaireimaginaire@boiteemail.fr \nDate :  28/12/2018 14:23:16 \nObjet :  Un objet \n\n"
empty_output = [
    {
        "text": "",
        "meta": "\n\nA :  societaireimaginaire@boiteemail.fr \nDate :  28/12/2018 14:23:16 \nObjet :  Un objet \n\n",
    }
]


def test_build_historic():
    input_df = pd.DataFrame([{"body": body}, {"body": empty_body}])
    print(input_df)
    output_df = pd.Series([output, empty_output])
    print(output_df)
    result = input_df.apply(build_historic, axis=1)
    pd.testing.assert_series_equal(result, output_df)
