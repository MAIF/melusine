# Melusine Regexes

Let's say you are a very busy fairy handling the organisation of a lot of fairy stuff.
You might want to apply a special king of magic named "regex" to your emails so that all the fairying goes smoothly.

There are some things you want sent directly to the trash:

* That annoying Voldemort dude keeps sending his `Avada Kedavra` thingy, which tingles everytime you open it, let's delete it on sight.
* Gandalf sends all the memes about him and his stupid catchphrase, you don't really care about it and want it deleted as well.

## Preparing our custom MelusineRegex

So you heat up your cauldron and start by cooking the following `MelusineRegex`

``` python hl_lines="6-11"
from melusine.base import MelusineRegex


class AnnoyingEmailsRegex(MelusineRegex):

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return dict(
            VOLDY_BEING_VOLDY="Avada Kedavra",
            GANDALF_BEING_GANDALF="You shall not pass",
        )

    @property  # (1)!
    def match_list(self) -> List[str]:
        return [
            "Avada Kedavra is a spell used by Lord Voldemort",
            "And then, you know me, I was not gonna let it pass so I told them : You shall not pass and obviously everyone clapped",
        ]

    @property  # (2)!
    def no_match_list(self) -> List[str]:
        return ["Abracadabra, here I am", "I told them not to pass"]
```

1. The code would not work without those, we'll see later how they are used
2. The code would not work without those, we'll see later how they are used


Let's use it on this incoming email.

``` python
to_delete_regex = AnnoyingEmailsRegex()
to_delete_detection = to_delete_regex(
    "You shall not pass through the magical portal of the lake from Monday to Thursday as it is currently under repair."
)

print(to_delete_detection[MelusineRegex.MATCH_RESULT])  # (1)!
print(to_delete_detection[MelusineRegex.POSITIVE_MATCH_FIELD])  # (2)!
```

1. Prints `True`
2. Prints `{'GANDALF_BEING_GANDALF': [{'start': 0, 'stop': 18, 'match_text': 'You shall not pass'}]}`

The thing is, our email actually came from the Magical Portal Society and was pretty important. We should exclude any email mentionning the magical portal from detection to make sure we still get the important infos.

## Using both negative and positive matches

``` python hl_lines="13-17"
from melusine.base import MelusineRegex


class AnnoyingEmailsRegex(MelusineRegex):

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

    @property  # (1)!
    def match_list(self) -> List[str]:
        return [
            "Avada Kedavra is a spell used by Lord Voldemort",
            "And then, you know me, I was not gonna let it pass so I told them : You shall not pass and obviously everyone clapped",
        ]

    @property  # (2)!
    def no_match_list(self) -> List[str]:
        return ["Abracadabra, here I am", "I told them not to pass"]
```

And now the Regex works as follows

``` python
to_delete_regex = AnnoyingEmailsRegex()
email = "You shall not pass through the magical portal of the lake from Monday to Thursday as it is currently under repair."
to_delete_detection = to_delete_regex(email)

print(to_delete_detection[MelusineRegex.MATCH_RESULT])  # (1)!
```

1. Prints `False`

## Preprocessing

Now, you know that old geezer of Gandalf, he tends to mispell words often.

So you want to make sure you detect any of his hobbit-feast-greasy-fingers-written emails too.

With a pinch of preprocessing, it's easy to do!

``` python
class PreMatchHookAnnoyingEmailsRegex(AnnoyingEmailsRegex):
    def pre_match_hook(self, text: str) -> str:
        text = text.replace("sholl not pass", "shall not pass")
        return text

    preprocessed_to_delete_regex = PreMatchHookAnnoyingEmailsRegex()
    spell_result = preprocessed_to_delete_regex.get_match_result(
        "Andthen,, I told Morgana 'You sholl not pass!' as she wanted topass... Im stil wonddering why she did not find it fundny..."
    )
```

## Using the regex result

As you can see, the negative cancels the positive, hence this email was not detected.
The complete `to_delete_detection` object looks like this:

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
spell_analysis_result: bool = to_delete_regex.get_match_result(email)  # (1)!
```
1. Returns `True`
Some of the older fairies might need a more detailed explanation on what triggered the deletion of an email.
Fortunately the describe methode serves this exact purpose.

``` python
to_delete_regex.describe(email)
```

Which will print:

>The MelusineRegex match result is : NEGATIVE
>
>The following text matched negatively: (PORTAL_MENTIONNED) magical portal of the lake
>
>The following text matched positively: (GANDALF_BEING_GANDALF) You shall not pass


## Examples list

Fairies social life can be hectic resulting in a variety of emails from all kinds of creatures.
Like most magical artefacts, regexes can be quite obscure and hard to decipher.
This is where the `match_list` and `no_match_list` properties come in handy:  

* Examples in the `match_list` should activate the MelusineRegex
* Examples in the `no_match_list` should not activate the MelusineRegex

The `test` method will be run at instanciation to check if the regex is working as intended.

``` python
from melusine.base import MelusineRegex


class AnnoyingEmailsRegex(MelusineRegex):

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return dict(
            VOLDY_BEING_VOLDY="Avada Kedavra",
            GANDALF_BEING_GANDALF="You shall not pass",
        )

    @property
    def match_list(self) -> List[str]:
        return [
            "Avada Kedavra is a spell used by Lord Voldemort", # (1)!
            "Erroneous example: This will not match!", # (2)!
        ]

    @property
    def no_match_list(self) -> List[str]:
        return ["Abracadabra, here I am", "I told them not to pass"]
        
regex = AnnoyingEmailsRegex() # (3)!
```

1. This example is aligned with the regex (activates a positive pattern and doesn't trigger any negative pattern).
2. This example is not aligned with the regex (doesn't activate any positive pattern).
3. This will raise an error as the second example in the `match_list` will not match.


## Even more advanced use case with "neutral"

As a fairy preserving the balance of the world and all, another case you would like to handle is differentiating your colleague Ifrit's emails between dangerous or not.
Contrarly to him you actually like the not-yet-totally-burning state of the world and would rather keep it that way.

But you cannot afford to go all-in everytime he jokingly sends false alarms emails. 

The good thing is that Ifrit is a bit of a dummy: 

* whevener he wants to burn the world for real and needs to be stopped he sends you an email with his intentions.
* if he is actually joking, the emails uses contractions which make his intentions super easy to guess (he is not just "a bit" of a dummy, he can be plain stupid sometimes).

That is were neutral regex can be of use. Whenever a neutral regex is matched, it is neutralized: all the match content is "blurred" and won't match anything later.


```python
class IfritAlertRegex(MelusineRegex):

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return dict(
            WORLD_MIGHT_BURN_1=r"see (the world|everything) (burn|in flames)",
            WORLD_MIGHT_BURN_2=r"make (the world|everything) (burn|in flames)",
        )

    @property
    def neutral(self) -> Union[str, Dict[str, str]]:
        return dict(
            WORLD_WONT_BURN_1=r"I wanna see (the world|everything) (burn|in flames)",
            WORLD_WONT_BURN_2=r"Imma make (the world|everything) (burn|in flames)",
            WORLD_WONT_BURN_3=r"I wanna make (the world|everything) (burn|in flames)",
        )

    @property
    def match_list(self) -> List[str]:
        return [
            "I want to see the world burn",
            "Let us make the world burn, shall we",
            "I wanna make the world burn and see everything in flames",  # (1)!
        ]

    @property
    def no_match_list(self) -> List[str]:
        return ["I wanna see the world burn", "Imma make everything in flames"]
```

1. In this specific case, neutral will come into action first and blur "I wanna make the world burn" but "see everything burn" will still match

## Conclusion

* The `MelusineRegex` class is a convenient tool to keep regexes clean, documented and easy to use.    
* Advanced features like "pre" and "post" match hooks bring flexibility to accommodate exotic use cases.  
* The `match_list` and `no_match_list` help onboard newcomers on what the regex does.  
* The `test` method is a great way to ensure the regex is working as intended.  

Now you can go back to your fairy duties and let the regex do the heavy lifting for you.
