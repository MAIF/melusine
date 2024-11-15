# Melusine Regexes

Let's say you are a very busy fairy handling the organisation of a lot of fairy stuff.
You might want to apply a special king of magic named "regex" to your emails so that all the fairying goes smoothly.

There are some things you want sent directly to the trash:

* That annoying Voldemort dude keeps sending his `Avada Kedavra` thingy, which tingles everytime you open it, let's delete it on sight.
* Gandalf sends all the memes about him and his stupid catchphrase, you don't really care about it and want it deleted as well.

## Preparing our custom MelusineRegex

So you heat up your cauldron and start by cooking the following `MelusineRegex`

``` python
from melusine.base import MelusineRegex

class SpellRegex(MelusineRegex):

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return dict(
            VOLDY_BEING_VOLDY="Avada Kedavra",
            GANDALF_BEING_GANDALF="You shall not pass",
            )

    @property# (1)!
    def match_list(self) -> List[str]:
        return [
            "Avada Kedavra is a spell used by Lord Voldemort",
            "And then, you know me, I was not gonna let it pass so I told them : You shall not pass and obviously everyone clapped",
        ]

    @property # (2)!
    def no_match_list(self) -> List[str]:
        return [
            "Abracadabra, here I am",
            "I told them not to pass"
        ]
```

1. The code would not work without those, we'll see later how they are used
2. The code would not work without those, we'll see later how they are used


Let's use it on the incoming emails.

``` python
spell_regex = SpellRegex()
spell_detection = spell_regex("You shall not pass through the magical portal of the lake from Monday to Thursday as it is currently under repair.")

print(spell_detection[MelusineRegex.MATCH_RESULT]) # (1)!
print(spell_detection[MelusineRegex.POSITIVE_MATCH_FIELD])# (2)! 
```

1. Prints `True`
2. Prints `{'GANDALF_BEING_GANDALF': [{'start': 0, 'stop': 18, 'match_text': 'You shall not pass'}]}`

The thing is, our email actually came from the Magical Portal Society and was pretty important. We should exclude any email mentionning the magical portal from detection to make sure we still get the important infos.

## Using both negative and positive matches

``` python
from melusine.base import MelusineRegex

class SpellRegex(MelusineRegex):

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return dict(
            VOLDY_BEING_VOLDY="Avada Kedavra",
            GANDALF_BEING_GANDALF="You shall not pass",
            )

    @property
    def negative(self) -> Union[str, Dict[str, str]]:
        return dict(
            PORTAL_MENTIONNED="magical portal of the lake",
            )

    @property# (1)!
    def match_list(self) -> List[str]:
        return [
            "Avada Kedavra is a spell used by Lord Voldemort",
            "And then, you know me, I was not gonna let it pass so I told them : You shall not pass and obviously everyone clapped",
        ]

    @property # (2)!
    def no_match_list(self) -> List[str]:
        return [
            "Abracadabra, here I am",
            "I told them not to pass"
        ]
```

And now the Regex works as follows

``` python
spell_regex = SpellRegex()
email = "You shall not pass through the magical portal of the lake from Monday to Thursday as it is currently under repair."
spell_detection = spell_regex(email)

print(spell_detection[MelusineRegex.MATCH_RESULT]) # (1)!
```

1. Prints `False`

## Preprocessing

Now, you know that old geezer of Gandalf, he tends to mispell words often.

So you want to make sure you detect any of his hobbit-feast-greasy-fingers-written emails too.

With a pinch of preprocessing, it's easy to do!

``` python
class PreMatchHookSpellRegex(SpellRegex):
    def pre_match_hook(self, text: str) -> str:
        text = text.replace("sholl not pass", "shall not pass")
        return text

    preprocessed_spell_regex = PreMatchHookSpellRegex()
    spell_result = preprocessed_spell_regex.get_match_result("Andthen,, I told Morgana 'You sholl not pass!' as she wanted topass... Im stil wonddering why she did not find it fundny...")
```

## Using the regex result

As you can see, the negative cancels the positive, hence this email was not detected.
The complete `spell_detection` object looks like this:

``` json
{
   "match_result":False,
   "neutral_match_data":{
   },
   "negative_match_data":{
      "PORTAL_MENTIONNED":[
         {
            "start":31,
            "stop":57,
            "match_text":"magical portal of the lake"
         }
      ]
   },
   "positive_match_data":{
      "GANDALF_BEING_GANDALF":[
         {
            "start":0,
            "stop":18,
            "match_text":"You shall not pass"
         }
      ]
   }
}
```

For a more straightforward approach, if you only need the regex result, you can use the following syntax:

``` python
spell_analysis_result: bool = spell_regex.get_match_result(email) # (1)!
```
1. Returns `True`
Some of the older fairies might need a more detailed explanation on what triggered the deletion of an email.
Fortunately the describe methode serves this exact purpose.

``` python
spell_regex.describe(email)
```

Which will print:

>The MelusineRegex match result is : NEGATIVE
>
>The following text matched negatively: (PORTAL_MENTIONNED) magical portal of the lake
>
>The following text matched positively: (GANDALF_BEING_GANDALF) You shall not pass


## Examples list

As we mentionned earlier, the MelusineRegex cannot work without examples of matches and no matches lists.

This makes the regex easier to share as the one developping it can offer examples and a newcomer on the project can quickly grasp the point of the regex.

``` python
spell_regex.match_list
```

