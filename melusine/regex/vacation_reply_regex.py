from typing import Dict, List, Optional, Union

from melusine.base import MelusineRegex


class VacationReplyRegex(MelusineRegex):
    """
    Detect vacation reply patterns such as "Je suis absent du bureau".
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return dict(
            VAC_REP_URGENCY=r"en\s+?cas\s+?d'?\s?urgenc.{1,100}(?:contact|app?eler?)|pour toute urgence.{1,100}contact",
            VAC_REP_HOLIDAYS=(
                r"^.{,30}(?:je suis |etant )?(:?actuellement)?(?:(?<!...\bete|.\betai[st]|etions|.\betiez)\sabsente?|"
                r"(?<!...\bete|.\betai[st]|etions|.\betiez)\sen cong[ée]s?)"
            ),
            VAC_REP_ON_MOVE=(
                r"(?<!serai\s|etais\s)en deplacement[_\s]prof|^.{1,15}(?<!...\bete|.\betai[st]|etions|.\betiez)"
                r"\sabsente?.{1,15}du.{1,10}flag_date_|^(?:absente?|en cong[ée]s?)"
            ),
            VAC_REP_OUT_OF_OFFICE=r"je prendra.{,7}connaissance",
            VAC_REP_OUT_OF_OFFICE_ENG=r"^.{,35}(?:i\s(?:am|will be)\s)?(:?currently\s)?(?:out of (?:the )?office)",
            VAC_REP_NO_REPLY=r"(r[eé]?ponse automatique|automatic reply|noreply)",
            VAC_REP_AUTO=r"cec[il] est (?:une r[ée]ponse|un (e-?)?mail) automatique",
            VAC_REP_ACKNOLEDGMENT=r"(?:^|par cet? e?mail,?\s)Nous\saccusons?\s(bonne\s)?r[ée]ception",
        )

    @property
    def neutral(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return dict(
            VAC_REP_FORBIDDEN_PROC=r"pas.{,10}traite|pas.{,10}identifie|pas.{,10}aboutir?",
            VAC_PREP_FORBIDDEN_ACTION=(
                r"dans.{,7}attente|merci par avance|(?:serai[st]?|aurai[st]?) souhait(able)?|relanc[ée]r?"
            ),
            VAC_PREP_FORBIDDEN_FORGOT=(
                r"je.{,10}perm(?:ets?|is?)|effectivement|\boublie?|malheureusement|suite [aà]|"
                r"(?:monsieur|madame).{,15}absent?"
            ),
            VAC_REP_FORBIDDEN_RETURN=(
                r"(?:retour|reponse) tardi(?:f|ve)|vien[ts] de (?:prendre (?:note|conn?aiss?ance)|recevoir)"
            ),
            VAC_REP_FORBIDDEN_EXTRA=(
                r"\bai\b.{,15}app?ell?er?|\bai\b.{,7}\blu[es]?\b|\barr?eter?.{,5}date|(?:aucune|pas).{,7}nouvelle|"
                r"chaque ann[ée]e|voisin"
            ),
            VAC_REP_FORBIDDEN_AMBIGUOUS=r"en\s+?cas\s+?de\s?besoin.{1,100}(?:contact|app?eler?)|going on leave",
            VAC_REP_FORBIDDEN_ABSENCE=(
                r"dur[ée]e? ind[ée]termin[ée]e?|"
                r"expertise judiciaire|doi.{,35}au plus tard|^.{,50}adresse[rz] vos|en mon absence"
            ),
            VAC_REP_FORBIDDEN_EMAIL=(
                r"pas prise? en compte|chang[ée]r?.{,5}(?:adresse|mail)|"
                r"mail.{,15}(?:(in)?acti|pas lu)|acc[eè]s limit[ée]"
            ),
            VAC_REP_FORBIDDEN_SICKNESS=r"arr?[eê]t de travail|\bmaternit[ée]\b|maladie",
            VAC_REP_FORBIDDEN_LEAVE=(
                r"\bposte\b.{,50}\bvacant\b|d[ée]finitivement|succe[ée]d[ée]|"
                r"quitt[ée]|remplacante?|plus parti.{,25}cabinet|retraite"
            ),
            VAC_REP_FORBIDDEN_HOME=(
                r"^.{,15}(?:devant ).{,5}(?:absente|en cong[ée]s?)|"
                r"^.{,15}absente? de (?:mon domicile?|chez|.{,5}maison)"
            ),
            VAC_REP_FORBIDDEN_CO=r"entreprise|document",
        )

    @property
    def match_list(self) -> List[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "en cas d'urgence, vous pouvez m'appeler sur mon mobile au 01 02 03 04 05",
            "pour toute urgence, vous pouvez me contacter au 01 02 03 04 05",
            "étant actuellement absent, je vous répondrais lors de mon retour le 01/01/2001",
            "Je suis en congé, je vous répondrais à mon retour",
            "Je suis absent du flag_date au flag_date",
            "je prendrais connaissance de votre message dès mon retour",
            "I am currently out of office",
            "ceci est une réponse automatique",
        ]

    @property
    def no_match_list(self) -> List[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "Je souhaite une réponse même si je suis actuellement en congé",
            "en attendant, je suis en congé, dans l'attente de votre réponse",
            "veuillez m'excuser, j'ai oublié de répondre avant de partir en vacances",
            "excusez mon retour tardif, j'étais en déplacement professionnel",
            "je suis absent pour une durée indéterminée, veuillez contacter Jane",
            "je suis actuellement en congé avec un accès limité à ma messagerie",
            "je suis en congé maladie",
            "je suis en congé maternité",
            "Je suis absent car je suis parti en retraite",
        ]
