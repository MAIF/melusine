import pandas as pd
from melusine.prepare_email.build_historic import build_historic

body = " \n  \n  \n Bonjours, \n  \n Suite a notre conversation téléphonique \
de Mardi , pourriez vous me dire la \n somme que je vous dois afin d'd'être \
en régularisation . \n  \n Merci bonne journée \n  \n Le mar. 22 mai 2018 \
à 10:20,  <gestionsocietaire@mutuelle.fr> a écrit\xa0: \n Bonjour. \n  \n \
Merci de bien vouloir prendre connaissance du document ci-joint : \n 1 - \
Relevé d'identité postal MUTUELLE (contrats) \n  \n Sentiments \
mutualistes. \n  \n La Mututelle \n  \n La visualisation des fichiers PDF \
nécessite Adobe Reader. \n  "


output = [
    {'text': " \n  \n  \n Bonjours, \n  \n Suite a notre conversation \
téléphonique de Mardi , pourriez vous me dire la \n somme que je vous \
dois afin d'd'être en régularisation . \n  \n Merci bonne journée",
     'meta': ''},
    {'text': " \n Bonjour. \n  \n Merci de bien vouloir prendre connaissance \
du document ci-joint : \n 1 - Relevé d'identité postal MUTUELLE \
(contrats) \n  \n Sentiments mutualistes. \n  \n La Mututelle \n  \n \
La visualisation des fichiers PDF nécessite Adobe Reader. \n  ",
     'meta': ' \n  \n Le mar. 22 mai 2018 à 10:20,  \
<gestionsocietaire@mutuelle.fr> a écrit\xa0:'}]


def test_build_historic():
    input_df = pd.DataFrame({
        'body': [body]
    })

    output_df = pd.Series([output])

    result = input_df.apply(build_historic, axis=1)
    pd.testing.assert_series_equal(result, output_df)
