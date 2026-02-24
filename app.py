import io
from pathlib import Path
import re
import unicodedata
import zipfile
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Analyse COPSOQ", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 500px at 10% -10%, #dbeafe 0%, transparent 60%),
                    radial-gradient(900px 400px at 90% -20%, #fde68a 0%, transparent 50%),
                    #f8fafc;
    }
    .block-container {padding-top: 1.4rem;}
    .kpi-card {
        border: 1px solid #d1d5db;
        border-radius: 14px;
        background: #ffffff;
        padding: 0.8rem 1rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }
    .small-muted {color: #475569; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ORDER_LEVELS = ["Tres faible", "Faible", "Modere", "Fort", "Tres fort"]
LEVEL_COLORS = {
    "Tres faible": "#90CAF9",
    "Faible": "#42A5F5",
    "Modere": "#66BB6A",
    "Fort": "#FFA726",
    "Tres fort": "#EF5350",
}
IMC_COLORS = {
    "Maigreur": "#6EC6FF",
    "Normal": "#66BB6A",
    "Surpoids": "#FFEE58",
    "Obesite": "#EF5350",
}

ITEM_MATCH_THRESHOLD = 0.60
PNG_CACHE_VERSION = "v2"

QUESTION_TEXT_MAP = {
    "Q1": "Prenez-vous du retard dans votre travail ?",
    "Q2": "Disposez-vous d'un temps suffisant pour accomplir vos taches professionnelles ?",
    "Q3": "Travaillez-vous a une cadence elevee tout au long de la journee ?",
    "Q4": "Est-il necessaire de maintenir un rythme soutenu au travail ?",
    "Q5": "Durant votre travail, devez-vous avoir l'oeil sur beaucoup de choses ?",
    "Q6": "Votre travail exige-t-il que vous vous souveniez de beaucoup de choses ?",
    "Q7": "Au travail, etes-vous informe(e) suffisamment a l'avance des decisions importantes, des changements ou de projets futurs ?",
    "Q8": "Recevez-vous toutes les informations dont vous avez besoin pour bien faire votre travail ?",
    "Q9": "Votre travail est-il reconnu et apprecie par le management ?",
    "Q10": "Etes-vous traite(e) equitablement au travail ?",
    "Q11": "Les conflits sont-ils resolus de maniere equitable ?",
    "Q12": "Le travail est-il reparti equitablement ?",
    "Q13": "Votre travail a-t-il des objectifs clairs ?",
    "Q14": "Savez-vous exactement ce que l'on attend de vous au travail ?",
    "Q15": "Au travail, etes-vous soumis(e) a des demandes contradictoires ?",
    "Q16": "Devez-vous parfois faire des choses qui auraient du etre faites autrement ?",
    "Q17": "Dans quelle mesure diriez-vous que votre superieur(e) hierarchique accorde une grande priorite a la satisfaction au travail ?",
    "Q18": "Dans quelle mesure diriez-vous que votre superieur(e) hierarchique est competent(e) dans la planification du travail ?",
    "Q19": "A quelle frequence votre superieur(e) hierarchique est-il (elle) dispose(e) a vous ecouter au sujet de vos problemes au travail ?",
    "Q20": "A quelle frequence recevez-vous de l'aide et du soutien de votre superieur(e) hierarchique ?",
    "Q21": "Le management fait-il confiance aux salaries quant a leur capacite a bien faire leur travail ?",
    "Q22": "Pouvez-vous faire confiance aux informations venant du management ?",
    "Q23": "Y a-t-il une bonne cooperation entre les collegues au travail ?",
    "Q24": "Dans l'ensemble, les salaries se font-ils confiance entre eux ?",
    "Q25": "A quelle frequence recevez-vous de l'aide et du soutien de vos collegues ?",
    "Q26": "A quelle frequence vos collegues se montrent-ils a l'ecoute de vos problemes au travail ?",
    "Q27": "Avez-vous une grande marge de manoeuvre dans votre travail ?",
    "Q28": "Pouvez-vous intervenir sur la quantite de travail qui vous est attribuee ?",
    "Q29": "Votre travail necessite-t-il que vous preniez des initiatives ?",
    "Q30": "Votre travail vous donne-il la possibilite d'apprendre des choses nouvelles ?",
    "Q31": "En general, diriez-vous que votre sante est :",
    "Q32": "A quelle frequence avez-vous ete irritable ?",
    "Q33": "A quelle frequence avez-vous ete stresse(e) ?",
    "Q34": "A quelle frequence vous etes-vous senti(e) a bout de force ?",
    "Q35": "A quelle frequence avez-vous ete emotionnellement epuise(e) ?",
    "Q36": "Votre travail vous place-t-il dans des situations destabilisantes sur le plan emotionnel ?",
    "Q37": "Votre travail est-il eprouvant sur le plan emotionnel ?",
    "Q38": "Sentez-vous que votre travail vous prend tellement d'energie que cela a un impact negatif sur votre vie privee ?",
    "Q39": "Sentez-vous que votre travail vous prend tellement de temps que cela a un impact negatif sur votre vie privee ?",
    "Q40": "Etes-vous inquiet(ete) a l'idee de perdre votre emploi ?",
    "Q41": "Craignez-vous d'etre mute(e) a un autre poste de travail contre votre volonte ?",
    "Q42": "Votre travail a-t-il du sens pour vous ?",
    "Q43": "Avez-vous le sentiment que le travail que vous faites est important ?",
    "Q44": "Recommanderiez-vous a un ami proche de postuler sur un emploi dans votre entreprise ?",
    "Q45": "Pensez-vous que votre entreprise est d'une grande importance pour vous ?",
    "Q46": "A quel point etes-vous satisfait(e) de votre travail dans son ensemble, en prenant en consideration tous les aspects ?",
}

SUBDOMAINS_CFG = {
    "charge_travail": ["Q1", "Q2"],
    "rythme_travail": ["Q3", "Q4"],
    "exigences_cognitives": ["Q5", "Q6"],
    "previsibilite": ["Q7", "Q8"],
    "reconnaissance": ["Q9", "Q10"],
    "equite": ["Q11", "Q12"],
    "clarte_roles": ["Q13", "Q14"],
    "conflit_roles": ["Q15", "Q16"],
    "qualite_leadership_superieur_hierarchique": ["Q17", "Q18"],
    "soutien_social_superieur_hierarchique": ["Q19", "Q20"],
    "confiance_salaries_management": ["Q21", "Q22"],
    "confiance_collegues": ["Q23", "Q24"],
    "soutien_social_collegues": ["Q25", "Q26"],
    "marge_manoeuvre": ["Q27", "Q28"],
    "possibilites_epanouissement": ["Q29", "Q30"],
    "sante_auto_evaluee": ["Q31"],
    "stress": ["Q32", "Q33"],
    "epuisement": ["Q34", "Q35"],
    "exigence_emotionnelle": ["Q36", "Q37"],
    "conflit_famille_travail": ["Q38", "Q39"],
    "insecurite_professionnelle": ["Q40", "Q41"],
    "sens_travail": ["Q42", "Q43"],
    "engagement_entreprise": ["Q44", "Q45"],
    "satisfaction_travail": ["Q46"],
}

SUBDOMAIN_LABELS_CFG = {
    "charge_travail": "Charge de travail_",
    "rythme_travail": "Rythme de travail_",
    "exigences_cognitives": "Exigences cognitives_",
    "previsibilite": "Prévisibilite_",
    "reconnaissance": "Reconnaissance_",
    "equite": "Équité_",
    "clarte_roles": "Clarté des rôles_ ",
    "conflit_roles": "Conflit de rôles_",
    "qualite_leadership_superieur_hierarchique": "Qualité du leadership_",
    "soutien_social_superieur_hierarchique": "Soutien du supérieur hiérarchique_",
    "confiance_salaries_management": "Confiance entre management et employé_",
    "confiance_collegues": "Confiance entre les collègues_",
    "soutien_social_collegues": "Soutien social de la part des collègues_",
    "marge_manoeuvre": "Marge de manœuvre_",
    "possibilites_epanouissement": "Possibilités d'épanouissement_",
    "sante_auto_evaluee": "Santé auto-évaluée_",
    "stress": "Stress_",
    "epuisement": "Epuisement_",
    "exigence_emotionnelle": "Exigences émotionnelles_",
    "conflit_famille_travail": "Conflit famille-travail_",
    "insecurite_professionnelle": "Insécurite professionnelle_",
    "sens_travail": "Sens du travail_",
    "engagement_entreprise": "Engagement dans l'entreprise_",
    "satisfaction_travail": "Satisfaction au travail_",
}

DOMAINS_CFG = {
    "Contraintes Quantitatives": ["charge_travail", "rythme_travail", "exigences_cognitives"],
    "Organisation et Leadership": [
        "previsibilite",
        "reconnaissance",
        "equite",
        "clarte_roles",
        "conflit_roles",
        "qualite_leadership_superieur_hierarchique",
        "soutien_social_superieur_hierarchique",
        "confiance_salaries_management",
    ],
    "Relations Horizontales": ["confiance_collegues", "soutien_social_collegues"],
    "Autonomie": ["marge_manoeuvre", "possibilites_epanouissement"],
    "Sante et Bien-être": [
        "sante_auto_evaluee",
        "stress",
        "epuisement",
        "exigence_emotionnelle",
        "conflit_famille_travail",
        "insecurite_professionnelle",
    ],
    "Vécu professionnel": ["sens_travail", "engagement_entreprise", "satisfaction_travail"],
}

DOMAIN_GROUPS = {
    "Contraintes Quantitatives": [
        (
            "Charge de travail_",
            "Prenez-vous du retard dans votre travail ?_Categorie",
            "Disposez-vous d'un temps suffisant pour accomplir vos taches professionnelles ?_Categorie",
        ),
        (
            "Rythme de travail_",
            "Travaillez-vous a une cadence elevee tout au long de la journee ?_Categorie",
            "Est-il necessaire de maintenir un rythme soutenu au travail ?_Categorie",
        ),
        (
            "Exigences cognitives_",
            "Durant votre travail, devez-vous avoir l'oeil sur beaucoup de choses ?_Categorie",
            "Votre travail exige-t-il que vous vous souveniez de beaucoup de choses ?_Categorie",
        ),
    ],
    "Organisation et Leadership": [
        (
            "Previsibilite_",
            "Au travail, etes-vous informe(e) suffisamment a l'avance au sujet par exemple de decisions importantes, de changements ou de projets futurs ?_Categorie",
            "Recevez-vous toutes les informations dont vous avez besoin pour bien faire votre travail ?_Categorie",
        ),
        (
            "Reconnaissance_",
            "Votre travail est-il reconnu et apprecie par le management ?_Categorie",
            "Etes-vous traite(e) equitablement au travail ?_Categorie",
        ),
        (
            "Equite_",
            "Les conflits sont-ils resolus de maniere equitable ?_Categorie",
            "Le travail est-il reparti equitablement ?_Categorie",
        ),
        (
            "Clarte des roles_",
            "Votre travail a-t-il des objectifs clairs ?_Categorie",
            "Savez-vous exactement ce que l'on attend de vous au travail ?_Categorie",
        ),
        (
            "Conflit de roles_",
            "Au travail, etes-vous soumis(e) a des demandes contradictoires ?_Categorie",
            "Devez-vous parfois faire des choses qui auraient du etre faites autrement ?_Categorie",
        ),
        (
            "Qualite du leadership_",
            "Dans quelle mesure diriez-vous que votre superieur(e) hierarchique accorde une grande priorite a la satisfaction au travail ?_Categorie",
            "Dans quelle mesure diriez-vous que votre superieur(e) hierarchique est competent(e) dans la planification du travail ?_Categorie",
        ),
        (
            "Soutien du superieur hierarchique_",
            "A quelle frequence votre superieur(e) hierarchique est-il (elle) dispose(e) a vous ecouter au sujet de vos problemes au travail ?_Categorie",
            "A quelle frequence recevez-vous de l'aide et du soutien de votre superieur(e) hierarchique ?_Categorie",
        ),
        (
            "Confiance management et salaries_",
            "Le management fait-il confiance aux salaries quant a leur capacite a bien faire leur travail ?_Categorie",
            "Pouvez-vous faire confiance aux informations venant du management ?_Categorie",
        ),
    ],
    "Relations Horizontales": [
        (
            "Confiance entre les collegues_",
            "Y a-t-il une bonne cooperation entre les collegues au travail ?_Categorie",
            "Dans l'ensemble, les salaries se font-ils confiance entre eux ?_Categorie",
        ),
        (
            "Soutien social de la part des collegues_",
            "A quelle frequence recevez-vous de l'aide et du soutien de vos collegues ?_Categorie",
            "A quelle frequence vos collegues se montrent-ils a l'ecoute de vos problemes au travail ?_Categorie",
        ),
    ],
    "Autonomie": [
        (
            "Marge de manoeuvre_",
            "Avez-vous une grande marge de manoeuvre dans votre travail ?_Categorie",
            "Pouvez-vous intervenir sur la quantite de travail qui vous est attribuee ?_Categorie",
        ),
        (
            "Possibilites d'epanouissement_",
            "Votre travail necessite-t-il que vous preniez des initiatives ?_Categorie",
            "Votre travail vous donne-il la possibilite d'apprendre des choses nouvelles ?_Categorie",
        ),
    ],
    "Sante et Bien-être": [
        (
            "Sante auto-evaluee_",
            "En general, diriez-vous que votre sante est :_Categorie",
            "A quelle frequence etes-vous (avez-vous ete) irritable ?_Categorie",
        ),
        (
            "Stress_",
            "A quelle frequence etes-vous (avez-vous ete) stresse(e) ?_Categorie",
            "A quelle frequence vous etes-vous senti(e) a bout de force ?_Categorie",
        ),
        (
            "Epuisement_",
            "A quelle frequence avez-vous ete emotionnellement epuise(e) ?_Categorie",
            "Votre travail vous place-t-il dans des situations destabilisantes sur le plan emotionnel ?_Categorie",
        ),
        (
            "Exigences emotionnelles_",
            "Votre travail est-il eprouvant sur le plan emotionnel ?_Categorie",
            "Sentez-vous que votre travail vous prend tellement d'energie que cela a un impact negatif sur votre vie privee ?_Categorie",
        ),
        (
            "Conflit famille-travail_",
            "Etes-vous inquiet(ete) a l'idee de perdre votre emploi ?_Categorie",
            "Craignez-vous d'etre mute(e) a un autre poste de travail contre votre volonte ?_Categorie",
        ),
        (
            "Insecurite professionnelle_",
            "Etes-vous inquiet(ete) a l'idee de perdre votre emploi ?_Categorie",
            "Craignez-vous d'etre mute(e) a un autre poste de travail contre votre volonte ?_Categorie",
        ),
    ],
    "Vécu professionnel": [
        (
            "Sens du travail_",
            "Votre travail a-t-il du sens pour vous ?_Categorie",
            "Avez-vous le sentiment que le travail que vous faites est important ?_Categorie",
        ),
        (
            "Engagement dans l'entreprise_",
            "Recommanderiez-vous a un ami proche de postuler sur un emploi dans votre entreprise ?_Categorie",
            "Pensez-vous que votre entreprise est d'une grande importance pour vous ?_Categorie",
        ),
        (
            "Satisfaction au travail_",
            "A quel point etes-vous satisfait(e) de votre travail dans son ensemble, en prenant en consideration tous les aspects ?_Categorie",
            "A quel point etes-vous satisfait(e) de votre travail dans son ensemble, en prenant en consideration tous les aspects ?_Categorie",
        ),
    ],
}

SOCIO_VARS = [
    "Tranche d'age",
    "Genre",
    "Situation matrimonial",
    "Catégorie IMC",
    "tabagisme",
    "Poste de travail",
    "Pratique reguliere du sport",
    "Consommation reguliere d'alcool",
    "Tranche ancienneté",
]

GENERAL_METRIC_ALIASES = {
    "Taille de l'effectif Total": [],
    "Nombre Homme": ["nombre_hommes", "nb_hommes", "hommes"],
    "Nombre de femme": ["nombre_femmes", "nb_femmes", "femmes"],
    "Age moyen de l'entreprise": ["age", "age moyen", "moyenne age"],
    "Taux de presense au travail": ["taux presence", "presence", "attendance"],
    "Taux d'absence pour maladie": ["absence maladie", "taux absence maladie", "sick leave"],
    "Taux d'absence non maladie": ["absence non maladie", "taux absence non maladie"],
    "Taux de depenses sante par employe": ["depenses sante", "health cost", "cout sante"],
    "Nombre d'accident du travail": ["accident travail", "nb accidents", "work accidents"],
    "Taux de productivite du travailleur": ["productivite", "worker productivity"],
}

RPS_SUBDOMAIN_COLUMNS = {
    "Charge de travail": "Charge de travail_",
    "Rythme de travail": "Rythme de travail_",
    "Exigence cognitive": "Exigences cognitives_",
    "Previsibilite": "Previsibilite_",
    "Reconnaissance": "Reconnaissance_",
    "Equite": "Equite_",
    "Clarte des roles": "Clarte des roles_",
    "Conflit de roles": "Conflit de roles_",
    "Qualite de leadership du superieur hierarchique": "Qualite du leadership_",
    "Soutien social de la part du superieur hierarchique": "Soutien du superieur hierarchique_",
    "Confiance entre les salaries et le management": "Confiance management et salaries_",
    "Confiance entre les collegues": "Confiance entre les collegues_",
    "Soutien social de la part des collegues": "Soutien social de la part des collegues_",
    "Marge de manoeuvre": "Marge de manoeuvre_",
    "Possibilitee d'epanouissement": "Possibilites d'epanouissement_",
    "Sante auto evaluee": "Sante auto-evaluee_",
    "Stress": "Stress_",
    "Epuisement": "Epuisement_",
    "Exigence emotionnelle (enmpathie collective)": "Exigences emotionnelles_",
    "Conflit Vie pro - Vie person": "Conflit famille-travail_",
    "Insecurite professionnelle": "Insecurite professionnelle_",
    "Sens du travail": "Sens du travail_",
    "Engagement dans l'entreprise": "Engagement dans l'entreprise_",
    "Satisfaction au travail": "Satisfaction au travail_",
}

RPS_GROUPS = {
    "Score de la perception de la charge globale de travail": [
        "Charge de travail",
        "Rythme de travail",
        "Exigence cognitive",
    ],
    "Score de la perception de l'Organisation et du leadership": [
        "Previsibilite",
        "Reconnaissance",
        "Equite",
        "Clarte des roles",
        "Conflit de roles",
        "Qualite de leadership du superieur hierarchique",
        "Soutien social de la part du superieur hierarchique",
        "Confiance entre les salaries et le management",
    ],
    "Perception des relations entre collegue (ou collaboration et esprits d'equipe)": [
        "Confiance entre les collegues",
        "Soutien social de la part des collegues",
    ],
    "Score de la perception de l'autonomie au travail": [
        "Marge de manoeuvre",
        "Possibilitee d'epanouissement",
    ],
    "Score de la perception de la sante et du bien-etre": [
        "Sante auto evaluee",
        "Stress",
        "Epuisement",
        "Exigence emotionnelle (enmpathie collective)",
        "Conflit Vie pro - Vie person",
        "Insecurite professionnelle",
    ],
    "Score de la perception du vecu professionnel": [
        "Sens du travail",
        "Engagement dans l'entreprise",
        "Satisfaction au travail",
    ],
}


def norm_text(value: str) -> str:
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def smart_find_column(columns: list[str], target: str) -> str | None:
    target_n = norm_text(target)
    lookup = {norm_text(c): c for c in columns}
    if target_n in lookup:
        return lookup[target_n]

    best_col = None
    best_score = 0.0
    for c in columns:
        col_n = norm_text(c)
        contains_bonus = 0.12 if target_n in col_n or col_n in target_n else 0.0
        score = SequenceMatcher(None, col_n, target_n).ratio() + contains_bonus
        if score > best_score:
            best_col = c
            best_score = score
    return best_col if best_score >= 0.78 else None


class CopsoqCleaner:
    def __init__(
        self,
        df: pd.DataFrame,
        drop_first_col: bool = True,
        fill_marital: bool = True,
        drop_indices: list[int] | None = None,
        missing_threshold: float = 0.51,
    ):
        self.raw_df = df.copy()
        self.cleaned_df = df.copy()
        self.drop_first_col = drop_first_col
        self.fill_marital = fill_marital
        self.drop_indices = drop_indices or []
        self.missing_threshold = missing_threshold
        self.removed_columns_missing: list[str] = []
        self.removed_columns_manual: list[str] = []
        self.cleaning_log = ""

    def _find_by_patterns(self, patterns: list[str]) -> str | None:
        for col in self.cleaned_df.columns:
            c = norm_text(col)
            if any(re.search(p, c) for p in patterns):
                return col
        return None

    def clean_common_variables(self):
        ops = []

        if self.drop_first_col and self.cleaned_df.shape[1] > 1:
            first_col = self.cleaned_df.columns[0]
            self.cleaned_df = self.cleaned_df.iloc[:, 1:]
            self.removed_columns_manual.append(str(first_col))
            ops.append(f"Premiere colonne supprimee: {first_col}")

        if self.drop_indices:
            valid_drop = [i for i in self.drop_indices if i in self.cleaned_df.index]
            if valid_drop:
                self.cleaned_df = self.cleaned_df.drop(valid_drop)
                ops.append(f"Lignes supprimees: {valid_drop}")

        if self.fill_marital:
            marital_col = self._find_by_patterns([r"situation", r"mari"])
            if marital_col is not None:
                self.cleaned_df[marital_col] = self.cleaned_df[marital_col].fillna("Non renseigne")
                ops.append(f"Situation matrimoniale completee: {marital_col}")

        missing_ratio = self.cleaned_df.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        if cols_to_drop:
            self.cleaned_df = self.cleaned_df.drop(columns=cols_to_drop)
        self.removed_columns_missing = [str(c) for c in cols_to_drop]
        if cols_to_drop:
            ops.append(
                "Colonnes > "
                + f"{self.missing_threshold * 100:.0f}% manquants supprimees: "
                + ", ".join(cols_to_drop)
            )

        age_col = self._find_by_patterns([r"\bage\b"])
        tranche_age_col = self._find_by_patterns([r"tranche.*age"])
        if age_col is not None and "Tranche d'age" not in self.cleaned_df.columns and tranche_age_col is None:
            age_vals = pd.to_numeric(self.cleaned_df[age_col], errors="coerce")
            self.cleaned_df["Tranche d'age"] = pd.cut(
                age_vals,
                bins=[0, 20, 30, 40, 50, 60, float("inf")],
                labels=["-20", "20-30", "31-40", "41-50", "51-60", "60+"],
                include_lowest=True,
            )
            ops.append(f"Tranche d'age creee depuis: {age_col}")
        elif "Tranche d'age" in self.cleaned_df.columns:
            ops.append("Tranche d'age existante conservee et utilisee")
        elif tranche_age_col is not None:
            self.cleaned_df["Tranche d'age"] = self.cleaned_df[tranche_age_col]
            ops.append(f"Tranche d'age existante detectee et alignee depuis: {tranche_age_col}")
        else:
            ops.append("Tranche d'age: colonnes age/tranche d'age non trouvees")

        anciennete_col = self._find_by_patterns([r"anciennete", r"anciennet"])
        if "Tranche ancienneté" in self.cleaned_df.columns:
            ops.append("Tranche ancienneté existante conservee")
        elif anciennete_col is not None:
            anc_vals = pd.to_numeric(self.cleaned_df[anciennete_col], errors="coerce")
            self.cleaned_df["Tranche ancienneté"] = pd.cut(
                anc_vals,
                bins=[0, 2, 5, 10, 20, float("inf")],
                labels=["0-2", "3-5", "6-10", "11-20", "21+"],
                include_lowest=True,
            )
            ops.append(f"Tranche ancienneté creee depuis: {anciennete_col}")
        else:
            ops.append("Tranche ancienneté: colonne anciennete non trouvee")

        imc_col = self._find_by_patterns([r"\bimc\b"])
        if imc_col is not None:
            self.cleaned_df = self.cleaned_df.drop(columns=[imc_col])
            self.removed_columns_manual.append(str(imc_col))
            ops.append(f"IMC existant supprime: {imc_col}")

        poids_col = self._find_by_patterns([r"\bpoids\b"])
        taille_col = self._find_by_patterns([r"\btaille\b"])
        if poids_col is not None and taille_col is not None:
            poids_vals = pd.to_numeric(self.cleaned_df[poids_col], errors="coerce")
            taille_vals = pd.to_numeric(self.cleaned_df[taille_col], errors="coerce")
            taille_positive = taille_vals[taille_vals > 0]
            if not taille_positive.empty and float(taille_positive.median()) > 3:
                taille_m = taille_vals / 100.0
            else:
                taille_m = taille_vals
            imc_vals = poids_vals / (taille_m**2)
            imc_vals = imc_vals.replace([float("inf"), float("-inf")], np.nan)
            self.cleaned_df["IMC"] = imc_vals
            self.cleaned_df["Catégorie IMC"] = pd.cut(
                self.cleaned_df["IMC"],
                bins=[0, 18.5, 25, 30, 200],
                labels=["Maigreur", "Normal", "Surpoids", "Obesite"],
                include_lowest=True,
            )
            ops.append(f"IMC/Catégorie IMC calcules depuis: poids={poids_col}, taille={taille_col}")

        self.cleaning_log = "Nettoyage COPSOQ applique:\n- " + "\n- ".join(ops) if ops else "Aucune operation appliquee."
        self.cleaned_df.attrs["cleaning_log"] = self.cleaning_log
        return self.cleaning_log


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def find_best_alias_column(columns: list[str], aliases: list[str], threshold: float = 0.74) -> str | None:
    if not aliases:
        return None
    best_col = None
    best = 0.0
    for col in columns:
        col_n = norm_text(col)
        for a in aliases:
            a_n = norm_text(a)
            contains_bonus = 0.12 if a_n in col_n or col_n in a_n else 0.0
            score = SequenceMatcher(None, col_n, a_n).ratio() + contains_bonus
            if score > best:
                best = score
                best_col = col
    return best_col if best >= threshold else None


def safe_mean(df: pd.DataFrame, col: str | None) -> float:
    if not col or col not in df.columns:
        return np.nan
    vals = to_numeric(df[col]).dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def safe_sum(df: pd.DataFrame, col: str | None) -> float:
    if not col or col not in df.columns:
        return np.nan
    vals = to_numeric(df[col]).dropna()
    return float(vals.sum()) if not vals.empty else np.nan


def safe_positive_rate(df: pd.DataFrame, col: str | None) -> float:
    if not col or col not in df.columns:
        return np.nan
    s = df[col].dropna()
    if s.empty:
        return np.nan

    num = to_numeric(s)
    if num.notna().sum() == len(s):
        vals = num.astype(float)
        if vals.max() <= 1.0 and vals.min() >= 0.0:
            return float(vals.mean() * 100.0)
        uniq = set(np.unique(vals.round(4)))
        if uniq.issubset({0.0, 1.0}):
            return float(vals.mean() * 100.0)
        return np.nan

    positives = {
        "oui",
        "yes",
        "true",
        "vrai",
        "fumeur",
        "fume",
        "smoker",
        "malade",
        "chronique",
        "handicap",
        "suivi",
        "psychologique",
        "1",
    }
    negatives = {"non", "no", "false", "faux", "aucun", "aucune", "jamais", "pas", "0"}

    mapped = []
    for v in s.astype(str):
        vn = norm_text(v)
        if not vn:
            continue
        words = vn.split()
        has_neg = vn in negatives or vn.startswith("non ") or "pas" in words or "jamais" in words
        has_pos = vn in positives or any(tok in words for tok in positives)

        if has_neg and has_pos:
            mapped.append(0.0)
            continue
        if has_neg:
            mapped.append(0.0)
            continue
        if has_pos:
            mapped.append(1.0)
            continue
        if any(tok in words for tok in negatives):
            mapped.append(0.0)

    if not mapped:
        return np.nan
    return float(np.mean(mapped) * 100.0)


def compute_general_test_metrics(df: pd.DataFrame):
    cols = list(df.columns)
    genre_col = smart_find_column(cols, "Genre")
    age_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Age moyen de l'entreprise"])
    presence_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Taux de presense au travail"])
    abs_mal_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Taux d'absence pour maladie"])
    abs_non_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Taux d'absence non maladie"])
    dep_sante_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Taux de depenses sante par employe"])
    accident_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Nombre d'accident du travail"])
    productivite_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Taux de productivite du travailleur"])

    if genre_col and genre_col in df.columns:
        g = df[genre_col].astype(str).str.lower().str.strip()
        nb_hommes = int(g.str.contains("hom", regex=False).sum())
        nb_femmes = int(g.str.contains("fem", regex=False).sum())
    else:
        nb_hommes_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Nombre Homme"])
        nb_femmes_col = find_best_alias_column(cols, GENERAL_METRIC_ALIASES["Nombre de femme"])
        nb_hommes = safe_sum(df, nb_hommes_col)
        nb_femmes = safe_sum(df, nb_femmes_col)

    metrics = {
        "Taille de l'effectif Total": float(len(df)),
        "Nombre Homme": nb_hommes,
        "Nombre de femme": nb_femmes,
        "Age moyen de l'entreprise": safe_mean(df, age_col),
        "Taux de presense au travail": safe_mean(df, presence_col),
        "Taux d'absence pour maladie": safe_mean(df, abs_mal_col),
        "Taux d'absence non maladie": safe_mean(df, abs_non_col),
        "Taux de depenses sante par employe": safe_mean(df, dep_sante_col),
        "Nombre d'accident du travail": safe_sum(df, accident_col),
        "Taux de productivite du travailleur": safe_mean(df, productivite_col),
    }
    source_map = {
        "Genre": genre_col,
        "Age": age_col,
        "Presence": presence_col,
        "Absence maladie": abs_mal_col,
        "Absence non maladie": abs_non_col,
        "Depenses sante": dep_sante_col,
        "Accidents": accident_col,
        "Productivite": productivite_col,
    }
    return metrics, source_map


def compute_rps_metrics(df: pd.DataFrame):
    level_to_score = {"Tres faible": 1, "Faible": 2, "Modere": 3, "Fort": 4, "Tres fort": 5}
    sub_scores = {}
    sources = {}
    for label, col_name in RPS_SUBDOMAIN_COLUMNS.items():
        found = smart_find_column(list(df.columns), col_name)
        sources[label] = found
        if not found:
            sub_scores[label] = np.nan
            continue
        s = df[found]
        if pd.api.types.is_numeric_dtype(s):
            vals = to_numeric(s).dropna()
        else:
            vals = s.map(level_to_score).dropna()
        sub_scores[label] = float(vals.mean()) if not vals.empty else np.nan

    group_scores = {}
    for g, items in RPS_GROUPS.items():
        vals = [sub_scores.get(item, np.nan) for item in items]
        group_scores[g] = float(np.nansum(vals)) if np.isfinite(np.nansum(vals)) else np.nan

    all_sub_vals = [v for v in sub_scores.values() if pd.notna(v)]
    global_score = float(np.sum(all_sub_vals)) if all_sub_vals else np.nan
    return sub_scores, group_scores, global_score, sources


def fmt_metric_value(label: str, val):
    if pd.isna(val):
        return "N/A"
    low = norm_text(label)
    if "taux" in low:
        if float(val) <= 1:
            return f"{float(val) * 100:.2f}%"
        return f"{float(val):.2f}%"
    if "nombre" in low or "effectif" in low:
        return f"{float(val):.2f}"
    return f"{float(val):.2f}"


def format_df_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(decimals)
    return out


def fmt_pct_label(value: float) -> str:
    v = round(float(value), 1)
    if float(v).is_integer():
        return f"{int(v)}%"
    return f"{v:.1f}%".replace(".", ",")


def pct_fontsize_for_block(value: float, min_size: int = 7, max_size: int = 11) -> int:
    v = float(value)
    if v < 3:
        return max(min_size, 7)
    if v < 5:
        return max(min_size, 8)
    if v < 8:
        return max(min_size, 9)
    if v < 12:
        return max(min_size, 10)
    return max_size


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.25)
    buf.seek(0)
    return buf.getvalue()


def get_cached_png_bytes(cache_key: str, fig, data_signature: str) -> bytes:
    store = st.session_state.setdefault("_png_cache", {})
    full_key = f"{PNG_CACHE_VERSION}|{cache_key}|{data_signature}"
    if full_key not in store:
        store[full_key] = fig_to_png_bytes(fig)
    return store[full_key]


def to_0_100_from_likert(score: float, min_val: float = 1.0, max_val: float = 5.0) -> float:
    if pd.isna(score):
        return np.nan
    clipped = min(max(float(score), min_val), max_val)
    return ((clipped - min_val) / (max_val - min_val)) * 100.0


def render_speed_gauge(title: str, score: float, height: int = 190):
    if pd.isna(score):
        st.info(f"{title}: non disponible")
        return
    gauge_val = to_0_100_from_likert(score)
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(gauge_val),
                number={"suffix": "%", "valueformat": ".2f"},
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f766e"},
                    "steps": [
                        {"range": [0, 20], "color": "#dbeafe"},
                        {"range": [20, 40], "color": "#bfdbfe"},
                        {"range": [40, 60], "color": "#fde68a"},
                        {"range": [60, 80], "color": "#fdba74"},
                        {"range": [80, 100], "color": "#fecaca"},
                    ],
                },
            )
        )
        fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.metric(title, f"{float(gauge_val):.2f}%")


@st.cache_data(show_spinner=False)
def load_file(file, sheet_name=0):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file, sheet_name=sheet_name)
    raise ValueError("Format non supporte")


@st.cache_data(show_spinner=False)
def load_file_from_bytes(file_bytes: bytes, file_name: str, sheet_name=0):
    name = file_name.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        return pd.read_csv(bio, sep=None, engine="python")
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(bio, sheet_name=sheet_name)
    raise ValueError("Format non supporte")


def prepare_data(
    df: pd.DataFrame,
    drop_first_col: bool,
    fill_marital: bool,
    drop_indices: list[int],
    missing_threshold: float,
):
    out = df.copy()
    if drop_first_col and out.shape[1] > 1:
        out = out.iloc[:, 1:]

    col_marital = smart_find_column(list(out.columns), "Situation matrimonial")
    if fill_marital and col_marital:
        out[col_marital] = out[col_marital].fillna("Non renseigne")

    valid_drop = [i for i in drop_indices if i in out.index]
    if valid_drop:
        out = out.drop(valid_drop)

    missing = out.isna().sum()
    max_na = int(np.floor(missing_threshold * len(out)))
    to_drop = missing[missing > max_na].index.tolist()
    if to_drop:
        out = out.drop(columns=to_drop)

    return out, to_drop, valid_drop


def get_imc_category(imc: float) -> str:
    if pd.isna(imc):
        return np.nan
    if imc < 18.5:
        return "Maigreur"
    if imc < 25:
        return "Normal"
    if imc < 30:
        return "Surpoids"
    return "Obesite"


def add_imc(df: pd.DataFrame):
    out = df.copy()
    poids_col = smart_find_column(list(out.columns), "Poids")
    taille_col = smart_find_column(list(out.columns), "Taille")
    if not poids_col or not taille_col:
        return out, None

    poids = to_numeric(out[poids_col])
    taille = to_numeric(out[taille_col])
    imc = poids / (taille / 100) ** 2
    out["IMC"] = imc.round(2)
    out["Catégorie IMC"] = out["IMC"].apply(get_imc_category)
    return out, (poids_col, taille_col)


def categoriser_auto(df: pd.DataFrame):
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    new_cols = {}

    for col in num_cols:
        vals = set(out[col].dropna().unique().tolist())
        if not vals:
            continue
        if not vals.issubset({1, 2, 3, 4, 5}):
            continue

        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)

        def classify(v):
            if pd.isna(v):
                return np.nan
            if v <= q1:
                return "Faible"
            if v <= q3:
                return "Modere"
            return "Fort"

        new_cols[f"{col}_Categorie"] = out[col].apply(classify)

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out, list(new_cols.keys())


def domaines(df: pd.DataFrame, nom_domaine: str, quest1: str, quest2: str):
    out = df.copy()
    q1_col = smart_find_column(list(out.columns), quest1)
    q2_col = smart_find_column(list(out.columns), quest2)
    if not q1_col or not q2_col:
        return out, (q1_col, q2_col)

    cross = {
        ("Faible", "Faible"): "Tres faible",
        ("Faible", "Modere"): "Faible",
        ("Faible", "Fort"): "Modere",
        ("Modere", "Faible"): "Faible",
        ("Modere", "Modere"): "Modere",
        ("Modere", "Fort"): "Fort",
        ("Fort", "Faible"): "Modere",
        ("Fort", "Modere"): "Fort",
        ("Fort", "Fort"): "Tres fort",
    }
    out[nom_domaine] = out.apply(lambda r: cross.get((r[q1_col], r[q2_col]), np.nan), axis=1)
    return out, (q1_col, q2_col)


def categorize_to_three_levels(series: pd.Series) -> pd.Series:
    num = to_numeric(series)
    if num.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)

    def classify(v):
        if pd.isna(v):
            return np.nan
        if v <= q1:
            return "Faible"
        if v <= q3:
            return "Modere"
        return "Fort"

    return num.apply(classify)


def categorize_to_five_levels(series: pd.Series) -> pd.Series:
    num = to_numeric(series)
    if num.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    q20 = num.quantile(0.20)
    q40 = num.quantile(0.40)
    q60 = num.quantile(0.60)
    q80 = num.quantile(0.80)

    def classify(v):
        if pd.isna(v):
            return np.nan
        if v <= q20:
            return "Tres faible"
        if v <= q40:
            return "Faible"
        if v <= q60:
            return "Modere"
        if v <= q80:
            return "Fort"
        return "Tres fort"

    return num.apply(classify)


def map_questions_to_columns(df: pd.DataFrame):
    cols = list(df.columns)
    q_map = {}
    for q, q_text in QUESTION_TEXT_MAP.items():
        target = norm_text(q_text)
        best_col = None
        best = 0.0
        for c in cols:
            c_n = norm_text(c)
            contains_bonus = 0.12 if target in c_n or c_n in target else 0.0
            score = SequenceMatcher(None, c_n, target).ratio() + contains_bonus
            if score > best:
                best = score
                best_col = c
        q_map[q] = best_col if best >= ITEM_MATCH_THRESHOLD else None
    return q_map


def build_domaines_from_question_config(df: pd.DataFrame):
    out = df.copy()
    q_map = map_questions_to_columns(out)
    q_cat_cols = {}

    for q, src_col in q_map.items():
        if not src_col or src_col not in out.columns:
            continue
        # Conserver les noms d'origine: pas de colonnes nommees Qx_*.
        src_cat = f"{src_col}_Categorie"
        if src_cat in out.columns:
            q_cat_cols[q] = src_cat
        else:
            out[src_cat] = categorize_to_three_levels(out[src_col])
            q_cat_cols[q] = src_cat

    created_subdomains = {}
    for sub_key, q_list in SUBDOMAINS_CFG.items():
        out_col = SUBDOMAIN_LABELS_CFG.get(sub_key, f"{sub_key}_")
        available = [q for q in q_list if q in q_cat_cols]
        if len(available) >= 2:
            q1 = q_cat_cols[available[0]]
            q2 = q_cat_cols[available[1]]
            out, found = domaines(out, out_col, q1, q2)
            if found[0] and found[1]:
                created_subdomains[sub_key] = out_col
        elif len(available) == 1:
            # Sous-domaine a item unique: auto-croisement pour garder la meme echelle finale.
            q1 = q_cat_cols[available[0]]
            out, found = domaines(out, out_col, q1, q1)
            if found[0]:
                created_subdomains[sub_key] = out_col

    domain_map = {}
    for domain_label, sub_keys in DOMAINS_CFG.items():
        cols = [created_subdomains[s] for s in sub_keys if s in created_subdomains]
        domain_map[domain_label] = cols

    return out, domain_map, q_map, created_subdomains


def build_domaines(df: pd.DataFrame):
    out = df.copy()
    created = {}
    missing_pairs = []

    for group, pairs in DOMAIN_GROUPS.items():
        created[group] = []
        for nom, q1, q2 in pairs:
            out, found = domaines(out, nom, q1, q2)
            if found[0] and found[1]:
                created[group].append(nom)
            else:
                missing_pairs.append((nom, q1, q2, found))
    return out, created, missing_pairs


def get_univariate_stats(series: pd.Series, n_rows: int):
    if pd.api.types.is_numeric_dtype(series):
        d = series.describe()
        stats = {
            "mean": float(d.get("mean", np.nan)),
            "std": float(d.get("std", np.nan)),
            "min": float(d.get("min", np.nan)),
            "25%": float(d.get("25%", np.nan)),
            "median": float(d.get("50%", np.nan)),
            "75%": float(d.get("75%", np.nan)),
            "max": float(d.get("max", np.nan)),
        }
    else:
        s = series.astype("object")
        mode = s.mode(dropna=True)
        mode_v = mode.iloc[0] if not mode.empty else "Aucun"
        freq_mode = int(s.value_counts(dropna=True).max()) if not s.value_counts(dropna=True).empty else 0
        stats = {
            "Unique": int(s.nunique(dropna=True)),
            "Mode": str(mode_v),
            "Freq Mode": int(freq_mode),
            "% Mode": float(round((freq_mode / n_rows) * 100, 2) if n_rows else np.nan),
        }

    freq = series.value_counts(dropna=False, normalize=True).mul(100).rename("Frequence (%)").reset_index()
    freq.columns = ["Categorie", "Frequence (%)"]
    freq["Categorie"] = freq["Categorie"].astype(str).replace("nan", "Valeurs manquantes")
    return stats, freq


def make_univariate_export(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        used = set()
        for col in df.columns:
            stats, freq = get_univariate_stats(df[col], len(df))
            name = re.sub(r"[\[\]:*?/\\]", "_", str(col))[:31]
            base = name
            i = 1
            while name.lower() in used:
                suffix = f"_{i}"
                name = (base[: 31 - len(suffix)] + suffix)
                i += 1
            used.add(name.lower())

            freq.to_excel(writer, sheet_name=name, index=False, startrow=0, startcol=0)
            pd.DataFrame(list(stats.items()), columns=["Statistique", "Valeur"]).to_excel(
                writer, sheet_name=name, index=False, startrow=0, startcol=4
            )
    output.seek(0)
    return output.getvalue()


def plot_distribution(df: pd.DataFrame, col: str):
    s = df[col]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 5:
        sns.histplot(s.dropna(), kde=True, ax=ax, color="#2563eb")
        ax.set_title(f"Histogramme - {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequence")
    else:
        fig.set_size_inches(8, 6)
        counts = s.dropna().value_counts()
        top = counts.head(5)
        if top.empty:
            ax.text(0.5, 0.5, "Aucune donnee", ha="center", va="center")
            ax.axis("off")
        else:
            values = top.values
            labels = top.index.astype(str)
            total = values.sum()
            small_pct_threshold = 6.0

            def pct_label(pct: float) -> str:
                return f"{pct:.2f}%" if pct >= small_pct_threshold else ""

            wedges, texts, autotexts = ax.pie(
                values,
                labels=None,
                autopct=pct_label,
                startangle=90,
                pctdistance=0.70,
                textprops={"fontsize": 10, "fontweight": "bold"},
            )

            small_slices = []
            for i, wedge in enumerate(wedges):
                pct = (values[i] / total) * 100 if total else 0
                if pct >= small_pct_threshold:
                    continue
                ang = (wedge.theta1 + wedge.theta2) / 2
                x = np.cos(np.deg2rad(ang))
                y = np.sin(np.deg2rad(ang))
                small_slices.append({"idx": i, "x": x, "y": y, "pct": pct})

            def spread_side(points: list[dict], min_gap: float = 0.20) -> dict[int, float]:
                if not points:
                    return {}
                y_min, y_max = -1.18, 1.18
                ordered = sorted(points, key=lambda p: p["y"])
                ys = []
                prev = y_min - min_gap
                for p in ordered:
                    y_val = max(p["y"] * 1.12, prev + min_gap)
                    ys.append(y_val)
                    prev = y_val
                overflow = ys[-1] - y_max
                if overflow > 0:
                    ys = [y - overflow for y in ys]
                    if ys[0] < y_min:
                        shift = y_min - ys[0]
                        ys = [y + shift for y in ys]
                return {p["idx"]: y for p, y in zip(ordered, ys)}

            right_side = [p for p in small_slices if p["x"] >= 0]
            left_side = [p for p in small_slices if p["x"] < 0]
            y_text_map = {**spread_side(right_side), **spread_side(left_side)}

            for p in small_slices:
                i = p["idx"]
                x = p["x"]
                y = p["y"]
                pct = p["pct"]
                ax.annotate(
                    f"{labels[i]} ({pct:.2f}%)",
                    xy=(x * 1.0, y * 1.0),
                    xytext=(0.5 * np.sign(x), y_text_map.get(i, 1.12 * y)),
                    ha="left" if x >= 0 else "right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#111827",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#6b7280",
                        "lw": 1.0,
                    },
                )

            for i, t in enumerate(autotexts):
                pct = (values[i] / total) * 100 if total else 0
                if pct >= small_pct_threshold:
                    t.set_text(f"{labels[i]}\n{pct:.2f}%")
                    t.set_fontsize(9)
                t.set_color("white")
            ax.legend(
                wedges,
                labels,
                title="Modalites",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
            ax.set_title(f"Repartition - {col}")
    fig.tight_layout()
    return fig


def plot_stacked_bar(df: pd.DataFrame, domain_cols: list[str], title: str):
    valid = [c for c in domain_cols if c in df.columns]
    if not valid:
        return None

    rows = []
    for c in valid:
        pct = df[c].value_counts(normalize=True).mul(100).reindex(ORDER_LEVELS, fill_value=0)
        rows.append(pct)
    dp = pd.DataFrame(rows, index=valid)

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5 * len(valid) + 1.5)))
    left = np.zeros(len(dp))
    y = np.arange(len(dp))

    for lvl in ORDER_LEVELS:
        vals = dp[lvl].values
        ax.barh(y, vals, left=left, color=LEVEL_COLORS[lvl], label=lvl)
        for i, v in enumerate(vals):
            if v > 0:
                if lvl == "Tres fort":
                    x = 101.0
                    ha = "left"
                    color = "#dc2626"
                    font_size = 11
                else:
                    x = left[i] + v / 2
                    ha = "center"
                    color = "white"
                    font_size = pct_fontsize_for_block(v)
                ax.text(
                    x,
                    i,
                    fmt_pct_label(v),
                    ha=ha,
                    va="center",
                    color=color,
                    fontsize=font_size,
                    fontweight="bold",
                )
        left += vals

    ax.set_yticks(y, dp.index)
    ax.set_xlim(0, 112)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("Pourcentage (%)")
    ax.set_title(title)
    ax.legend(ncol=5, bbox_to_anchor=(0.5, -0.18), loc="upper center", frameon=False)
    fig.tight_layout()
    return fig


def bivariate_table(df: pd.DataFrame, var1: str, var2: str):
    def round_row_to_100(row: pd.Series, decimals: int = 2) -> pd.Series:
        scale = 10**decimals
        target = int(100 * scale)
        raw = row.values.astype(float) * scale
        floored = np.floor(raw).astype(int)
        remainder = target - floored.sum()

        if remainder > 0:
            order = np.argsort(-(raw - floored))
            for idx in order[:remainder]:
                floored[idx] += 1
        elif remainder < 0:
            order = np.argsort(raw - floored)
            for idx in order[: abs(remainder)]:
                floored[idx] -= 1

        return pd.Series(floored / scale, index=row.index)

    temp = df[[var1, var2]].dropna()
    if temp.empty:
        return None
    ct = pd.crosstab(temp[var1], temp[var2], normalize="index").mul(100)
    ct = ct.reindex(columns=ORDER_LEVELS, fill_value=0)
    ct = ct.apply(round_row_to_100, axis=1)
    ct["Total"] = ct.sum(axis=1).round(2)
    return ct




def build_bivariate_figure(df: pd.DataFrame, socio_col: str, out_col: str):
    ct = bivariate_table(df, socio_col, out_col)
    if ct is None:
        return None, None

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5 * len(ct) + 1.5)))
    left = np.zeros(len(ct))
    y = np.arange(len(ct))
    for lvl in ORDER_LEVELS:
        vals = ct[lvl].values
        ax.barh(y, vals, left=left, color=LEVEL_COLORS[lvl], label=lvl)
        for i, v in enumerate(vals):
            if v > 0:
                if lvl == "Tres fort":
                    x = 101.0
                    ha = "left"
                    color = "#dc2626"
                    font_size = 11
                else:
                    x = left[i] + v / 2
                    ha = "center"
                    color = "white"
                    font_size = pct_fontsize_for_block(v)
                ax.text(
                    x,
                    i,
                    fmt_pct_label(v),
                    ha=ha,
                    va="center",
                    color=color,
                    fontsize=font_size,
                    fontweight="bold",
                )
        left += vals
    ax.set_yticks(y, ct.index.astype(str))
    ax.set_xlim(0, 112)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("Pourcentage (%)")
    ax.set_title(f"Répartition des employés de Cofina - {socio_col} selon {out_col}")
    ax.legend(ncol=5, bbox_to_anchor=(0.5, -0.18), loc="upper center", frameon=False)
    fig.tight_layout()
    return fig, ct


def export_bivariate_graphs_zip(df: pd.DataFrame, socio_candidates: list[str], outcome_cols: list[str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        exported = 0
        for socio_col in socio_candidates:
            if socio_col not in df.columns:
                continue
            for out_col in outcome_cols:
                if out_col not in df.columns:
                    continue
                fig, ct = build_bivariate_figure(df, socio_col, out_col)
                if fig is None or ct is None:
                    continue
                socio_name = re.sub(r"[\\/*?:\"<>|]", "_", str(socio_col))
                out_name = re.sub(r"[\\/*?:\"<>|]", "_", str(out_col))
                png_bytes = fig_to_png_bytes(fig)
                zf.writestr(f"croisements_bivaries/{socio_name}_x_{out_name}.png", png_bytes)
                plt.close(fig)
                exported += 1

        if exported == 0:
            zf.writestr("croisements_bivaries/README.txt", "Aucun croisement bivarié exploitable.")

    output.seek(0)
    return output.getvalue()
def export_bivariate_excel(df: pd.DataFrame, outcome_cols: list[str]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        idx = 0
        for s in SOCIO_VARS:
            s_col = smart_find_column(list(df.columns), s)
            if not s_col:
                continue
            for out in outcome_cols:
                if out not in df.columns:
                    continue
                ct = bivariate_table(df, s_col, out)
                if ct is None:
                    continue
                name = f"{s_col[:12]}_x_{out[:12]}".replace(" ", "_")[:31]
                ct.to_excel(writer, sheet_name=name)
                idx += 1
        if idx == 0:
            pd.DataFrame({"Info": ["Aucun tableau genere"]}).to_excel(writer, sheet_name="vide", index=False)
    output.seek(0)
    return output.getvalue()


def export_base_with_tests_excel(raw_df: pd.DataFrame, prepared: pd.DataFrame, domain_map: dict, outcome_cols: list[str]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        used_sheet_names = set()

        def unique_sheet_name(name: str) -> str:
            safe = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip() or "Feuille"
            safe = safe[:31]
            base = safe
            i = 1
            while safe.lower() in used_sheet_names:
                suffix = f"_{i}"
                safe = (base[: 31 - len(suffix)] + suffix)
                i += 1
            used_sheet_names.add(safe.lower())
            return safe

        raw_df.to_excel(writer, sheet_name=unique_sheet_name("Base_brute"), index=False)
        prepared.to_excel(writer, sheet_name=unique_sheet_name("Base_preparee"), index=False)

        uni_rows = []
        for col in prepared.columns:
            stats, _ = get_univariate_stats(prepared[col], len(prepared))
            row = {"Variable": col}
            row.update(stats)
            uni_rows.append(row)
        pd.DataFrame(uni_rows).to_excel(writer, sheet_name=unique_sheet_name("Test_univarie"), index=False)

        biv_rows = []
        for s in SOCIO_VARS:
            s_col = smart_find_column(list(prepared.columns), s)
            if not s_col:
                continue
            for out in outcome_cols:
                if out not in prepared.columns:
                    continue
                ct = bivariate_table(prepared, s_col, out)
                if ct is None:
                    continue
                temp = ct.reset_index().rename(columns={ct.index.name or "index": "Modalite"})
                temp.insert(0, "Outcome", out)
                temp.insert(0, "Covariable", s_col)
                biv_rows.append(temp)
        if biv_rows:
            pd.concat(biv_rows, ignore_index=True).to_excel(
                writer,
                sheet_name=unique_sheet_name("Test_bivarie"),
                index=False,
            )
        else:
            pd.DataFrame({"Info": ["Aucun tableau bivarie genere"]}).to_excel(
                writer, sheet_name=unique_sheet_name("Test_bivarie"), index=False
            )

        domain_rows = []
        for group, cols in domain_map.items():
            for c in cols:
                if c not in prepared.columns:
                    continue
                pct = prepared[c].value_counts(normalize=True).mul(100).reindex(ORDER_LEVELS, fill_value=0)
                row = {"Groupe": group, "Sous-domaine": c}
                for lvl in ORDER_LEVELS:
                    row[lvl] = float(pct[lvl])
                row["Total"] = float(sum(row[lvl] for lvl in ORDER_LEVELS))
                domain_rows.append(row)
        if domain_rows:
            pd.DataFrame(domain_rows).to_excel(writer, sheet_name=unique_sheet_name("Domaines_COPSOQ"), index=False)
        else:
            pd.DataFrame({"Info": ["Aucun domaine disponible"]}).to_excel(
                writer, sheet_name=unique_sheet_name("Domaines_COPSOQ"), index=False
            )

    output.seek(0)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def run_analysis_pipeline(raw_df: pd.DataFrame):
    cleaner = CopsoqCleaner(
        raw_df,
        drop_first_col=False,
        fill_marital=True,
        drop_indices=[],
        missing_threshold=0.51,
    )
    cleaner.clean_common_variables()
    prepared = cleaner.cleaned_df.copy()

    # Fallback IMC si les patterns du cleaner n'ont pas abouti.
    prepared, _ = add_imc(prepared)
    prepared, cat_cols = categoriser_auto(prepared)
    prepared_cfg, domain_map_cfg, question_map, _ = build_domaines_from_question_config(prepared)
    matched_q_count = sum(1 for v in question_map.values() if v is not None)

    if matched_q_count >= 8 and sum(len(v) for v in domain_map_cfg.values()) > 0:
        prepared = prepared_cfg
        domain_map = domain_map_cfg
        missing_domains = []
    else:
        prepared, domain_map, missing_domains = build_domaines(prepared)

    outcome_cols = []
    for _, cols in domain_map.items():
        outcome_cols.extend(cols)
    outcome_cols = list(dict.fromkeys(outcome_cols))

    general_metrics, general_sources = compute_general_test_metrics(prepared)
    rps_sub_scores, rps_group_scores, rps_global_score, rps_sources = compute_rps_metrics(prepared)

    return {
        "prepared": prepared,
        "cat_cols": cat_cols,
        "domain_map": domain_map,
        "missing_domains": missing_domains,
        "matched_q_count": matched_q_count,
        "outcome_cols": outcome_cols,
        "general_metrics": general_metrics,
        "general_sources": general_sources,
        "rps_sub_scores": rps_sub_scores,
        "rps_group_scores": rps_group_scores,
        "rps_global_score": rps_global_score,
        "rps_sources": rps_sources,
        "cleaning_log": cleaner.cleaning_log,
    }


st.title("Analyse COPSOQ")
st.markdown("<p class='small-muted'>Pipeline complet: preparation, univariee, domaines, bivariee et exports.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Parametres")
    uploaded = st.file_uploader("Fichier de donnees (Excel/CSV)", type=["xlsx", "xls", "xlsm", "csv"])
    st.caption("Affichage KPI uniquement (sans jauges).")

if not uploaded:
    st.info("Importez votre fichier pour lancer l'analyse.")
    st.stop()

try:
    raw_df = load_file_from_bytes(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"Lecture impossible: {exc}")
    st.stop()
pipeline = run_analysis_pipeline(raw_df)
prepared = pipeline["prepared"]
cat_cols = pipeline["cat_cols"]
domain_map = pipeline["domain_map"]
missing_domains = pipeline["missing_domains"]
matched_q_count = pipeline["matched_q_count"]
outcome_cols = pipeline["outcome_cols"]
general_metrics = pipeline["general_metrics"]
general_sources = pipeline["general_sources"]
rps_sub_scores = pipeline["rps_sub_scores"]
rps_group_scores = pipeline["rps_group_scores"]
rps_global_score = pipeline["rps_global_score"]
rps_sources = pipeline["rps_sources"]
cleaning_log = pipeline["cleaning_log"]
data_signature = f"{uploaded.name}|{prepared.shape[0]}|{prepared.shape[1]}|{int(prepared.isna().sum().sum())}"
base_name = re.sub(r"\.[^.]+$", "", uploaded.name)
base_name = re.sub(r"[\\/*?:\"<>|]", "_", base_name)


tab0, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard executif", "Analyse univariee", "Domaines COPSOQ", "Analyse bivariee", "Exports"]
)

with tab0:
    
    cols_raw = list(raw_df.columns)
    cols_prepared = list(prepared.columns)

    genre_col = smart_find_column(cols_prepared, "Genre") or smart_find_column(cols_raw, "Genre")
    if genre_col and genre_col in prepared.columns:
        g = prepared[genre_col].astype(str).str.lower().str.strip()
    elif genre_col and genre_col in raw_df.columns:
        g = raw_df[genre_col].astype(str).str.lower().str.strip()
    else:
        g = pd.Series(dtype="object")

    nb_hommes = int(g.str.contains("hom", regex=False).sum()) if not g.empty else np.nan
    nb_femmes = int(g.str.contains("fem", regex=False).sum()) if not g.empty else np.nan
    effectif_total = int(len(raw_df))

    age_col = find_best_alias_column(cols_raw, GENERAL_METRIC_ALIASES["Age moyen de l'entreprise"])
    age_mean = safe_mean(raw_df, age_col)
    tranche_age_col = smart_find_column(cols_raw, "Tranche d'age") or smart_find_column(cols_prepared, "Tranche d'age")
    if pd.notna(age_mean):
        age_display = f"{int(round(age_mean))}"
    elif tranche_age_col and tranche_age_col in raw_df.columns:
        ta_mode = raw_df[tranche_age_col].dropna().astype(str).mode()
        age_display = str(ta_mode.iloc[0]) if not ta_mode.empty else "N/A"
    elif tranche_age_col and tranche_age_col in prepared.columns:
        ta_mode = prepared[tranche_age_col].dropna().astype(str).mode()
        age_display = str(ta_mode.iloc[0]) if not ta_mode.empty else "N/A"
    else:
        age_display = "N/A"

    tabag_col = smart_find_column(cols_raw, "tabagisme") or smart_find_column(cols_prepared, "tabagisme")
    mal_chron_col = smart_find_column(cols_raw, "maladie chronique") or smart_find_column(
        cols_prepared, "maladie chronique"
    )
    handicap_col = smart_find_column(cols_raw, "handicap physique") or smart_find_column(
        cols_prepared, "handicap physique"
    )
    suivi_psy_col = smart_find_column(cols_raw, "Avez-vous été suivi pour un probleme psychologique")

    taux_tabag = safe_positive_rate(raw_df, tabag_col) if tabag_col in raw_df.columns else safe_positive_rate(prepared, tabag_col)
    taux_mal_chron = (
        safe_positive_rate(raw_df, mal_chron_col)
        if mal_chron_col in raw_df.columns
        else safe_positive_rate(prepared, mal_chron_col)
    )
    taux_handicap = (
        safe_positive_rate(raw_df, handicap_col) if handicap_col in raw_df.columns else safe_positive_rate(prepared, handicap_col)
    )
    taux_suivi_psy = (
        safe_positive_rate(raw_df, suivi_psy_col)
        if suivi_psy_col in raw_df.columns
        else safe_positive_rate(prepared, suivi_psy_col)
    )

    st.markdown("### Données Générales du test")
    cols_gen = st.columns(3)
    cols_gen[0].metric("Taille de l'effectif Total", f"{effectif_total}")
    cols_gen[1].metric("Nombre Homme", f"{int(nb_hommes)}" if pd.notna(nb_hommes) else "N/A")
    cols_gen[2].metric("Nombre de femme", f"{int(nb_femmes)}" if pd.notna(nb_femmes) else "N/A")

    cols_gen2 = st.columns(3)
    cols_gen2[0].metric("Age moyen de l'entreprise", age_display)
    cols_gen2[1].metric("Taux de tabagisme", fmt_metric_value("Taux", taux_tabag))
    cols_gen2[2].metric("Taux de maladie chrhronique", fmt_metric_value("Taux", taux_mal_chron))

    cols_gen3 = st.columns(3)
    cols_gen3[0].metric("Taux d'handicp physique", fmt_metric_value("Taux", taux_handicap))
    cols_gen3[1].metric("Taux de suivi psychologique", fmt_metric_value("Taux", taux_suivi_psy))

    with st.expander("Journal de nettoyage automatique", expanded=False):
        st.text(cleaning_log)
        st.write(f"Questions COPSOQ detectees automatiquement: {matched_q_count}/46")

    st.markdown("### Données Générales RPS")
    def sum_raw_targets(df_src: pd.DataFrame, targets: list[str]) -> float:
        total = 0.0
        found_any = False
        used_cols = set()
        for target in targets:
            found = smart_find_column(list(df_src.columns), target)
            if found and found not in used_cols:
                sval = safe_sum(df_src, found)
                if pd.notna(sval):
                    total += float(sval)
                    found_any = True
                used_cols.add(found)
        return total if found_any else np.nan

    kpi_rps = [
        ("Perception de la charge globale de travail", ["Charge de travail"]),
        (
            "Perception de l'Organisation et du leadership",
            [
                "Previsibilite",
                "Reconnaissance",
                "Equite",
                "Clarte des roles",
                "Conflit de roles",
                "Qualite de leadership du superieur hierarchique",
                "Soutien social de la part du superieur hierarchique",
                "Confiance entre les salaries et le management",
            ],
        ),
        (
            "Perception des relations entre collègue",
            ["Confiance entre les collegues", "Soutien social de la part des collegues"],
        ),
        ("Perception de l'autonomie au travail", ["Marge de manoeuvre", "Possibilitee d'epanouissement"]),
        (
            "Perception de la santé et du bien-être",
            [
                "Sante auto evaluee",
                "Stress",
                "Epuisement",
                "Exigence emotionnelle",
                "Conflit Vie pro - Vie person",
                "Insecurite professionnelle",
            ],
        ),
        ("Perception du vécu professionnel", ["Sens du travail", "Engagement dans l'entreprise", "Satisfaction au travail"]),
        ("Stress", ["Stress"]),
        ("Épuisement", ["Epuisement"]),
        ("Sens du travail", ["Sens du travail"]),
        ("Santé auto évaluée", ["Sante auto evaluee"]),
        ("Exigences émotionnelles", ["Exigence emotionnelle"]),
        ("Conflit famille/travail", ["Conflit Vie pro - Vie person", "Conflit famille-travail"]),
    ]
    cols_rps = st.columns(3)
    for i, (label, targets) in enumerate(kpi_rps):
        total_val = sum_raw_targets(raw_df, targets)
        cols_rps[i % 3].metric(label, f"{int(round(total_val))}" if pd.notna(total_val) else "N/A")

with tab2:
    st.subheader("Statistiques univariees")
    col = st.selectbox("Variable", prepared.columns.tolist())
    series_for_univariate = prepared[col]
    recomputed_five = False
    if str(col).endswith("_Categorie"):
        base_col = str(col).rsplit("_Categorie", 1)[0]
        if base_col in prepared.columns:
            base_num = to_numeric(prepared[base_col])
            if base_num.notna().sum() > 0:
                series_for_univariate = categorize_to_five_levels(base_num)
                recomputed_five = True

    stats, freq = get_univariate_stats(series_for_univariate, len(prepared))

    left, right = st.columns([1, 1.4])
    with left:
        if recomputed_five:
            st.caption("Variable _Categorie recalculee en 5 niveaux: Tres faible, Faible, Modere, Fort, Tres fort.")
        st.write("Statistiques")
        st.dataframe(
            format_df_for_display(pd.DataFrame(list(stats.items()), columns=["Mesure", "Valeur"])),
            use_container_width=True,
        )
        st.write("Frequences detaillees")
        st.dataframe(format_df_for_display(freq), use_container_width=True, height=280)
    with right:
        temp_df = prepared.copy()
        temp_df[col] = series_for_univariate
        fig = plot_distribution(temp_df, col)
        _, bcol = st.columns([8, 1])
        with bcol:
            svg_name = re.sub(r"[\\/*?:\"<>|]", "_", str(col))
            st.download_button(
                "PNG",
                data=get_cached_png_bytes(f"dl_png_uni_{svg_name}", fig, data_signature),
                file_name=f"{svg_name}.png",
                mime="image/png",
                key=f"dl_png_uni_{svg_name}",
            )
        st.pyplot(fig, use_container_width=True)

with tab3:
    st.subheader("Domaines et sous-domaines")
    domain_choices = [k for k, v in domain_map.items() if v]
    if not domain_choices:
        st.warning("Aucun domaine exploitable detecte.")
        domain_choices = list(domain_map.keys()) if domain_map else []
    grp = st.selectbox("Groupe", domain_choices) if domain_choices else None
    cols = domain_map.get(grp, []) if grp else []

    if not cols:
        st.warning("Aucun domaine construit pour ce groupe avec les colonnes actuelles.")
    else:
        fig = plot_stacked_bar(
            prepared,
            cols,
            f"Répartition des employés de Cofina selon {grp} au travail",
        )
        if fig:
            _, bcol = st.columns([8, 1])
            with bcol:
                grp_name = re.sub(r"[\\/*?:\"<>|]", "_", str(grp))
                st.download_button(
                    "PNG",
                    data=get_cached_png_bytes(f"dl_png_domain_{grp_name}", fig, data_signature),
                    file_name=f"{grp_name}.png",
                    mime="image/png",
                    key=f"dl_png_domain_{grp_name}",
                )
            st.pyplot(fig, use_container_width=True)

    if missing_domains:
        with st.expander("Paires non construites (questions introuvables)"):
            warn_df = pd.DataFrame(
                [
                    {
                        "Domaine": d,
                        "Q1 trouvee": bool(f[0]),
                        "Q2 trouvee": bool(f[1]),
                    }
                    for d, _, _, f in missing_domains
                ]
            )
            st.dataframe(warn_df, use_container_width=True)

    st.markdown("### Domaines et sous-domaines croiser")
    socio_candidates_tab3 = [smart_find_column(list(prepared.columns), x) for x in SOCIO_VARS]
    socio_candidates_tab3 = [x for x in socio_candidates_tab3 if x]
    domain_choices_cross = [k for k, v in domain_map.items() if v]
    if not domain_choices_cross or not socio_candidates_tab3:
        st.info("Selection indisponible: groupe/sous-domaines/socio-vars insuffisants.")
    else:
        c_cross_1, c_cross_2, c_cross_3 = st.columns(3)
        grp_cross = c_cross_1.selectbox("Groupe (les domaines)", domain_choices_cross, key="tab3_grp_cross")
        socio_cross_col = c_cross_2.selectbox(
            "Variable socio-demographique",
            ["Aucune"] + socio_candidates_tab3,
            key="tab3_socio",
        )
        if socio_cross_col == "Aucune":
            socio_modalites = []
            socio_choice = c_cross_3.selectbox("Modalite", ["Vue globale"], key="tab3_socio_modalite")
        else:
            socio_modalites = prepared[socio_cross_col].dropna().astype(str).unique().tolist()
            socio_modalites = sorted(socio_modalites)
            socio_choice = c_cross_3.selectbox(
                "Modalite",
                ["Toutes"] + socio_modalites,
                key="tab3_socio_modalite",
            )
        cross_cols = domain_map.get(grp_cross, [])
        if not cross_cols:
            st.warning("Aucun sous-domaine disponible pour ce groupe.")
        else:
            if socio_cross_col == "Aucune":
                modalities_to_show = [None]
            elif socio_choice == "Toutes":
                modalities_to_show = socio_modalites
            else:
                modalities_to_show = [socio_choice]

            if not modalities_to_show:
                st.warning("Aucune modalite disponible.")
            else:
                for mod in modalities_to_show:
                    if mod is None:
                        df_cross = prepared.copy()
                    else:
                        df_cross = prepared[prepared[socio_cross_col].astype(str) == mod].copy()
                    if df_cross.empty:
                        continue

                    if mod is None:
                        st.markdown("**Vue globale (aucun filtre socio-demographique)**")
                    else:
                        st.markdown(f"**{socio_cross_col} = {mod}**")
                    fig_cross = plot_stacked_bar(
                        df_cross,
                        cross_cols,
                        f"Répartition des employés de Cofina selon {grp_cross} du travail",
                    )
                    if fig_cross:
                        _, bcol = st.columns([8, 1])
                        with bcol:
                            grp_name = re.sub(r"[\\/*?:\"<>|]", "_", str(grp_cross))
                            mod_name = re.sub(r"[\\/*?:\"<>|]", "_", str(mod if mod is not None else "Aucune"))
                            socio_name = re.sub(
                                r"[\\/*?:\"<>|]", "_", str(socio_cross_col if socio_cross_col != "Aucune" else "Aucune")
                            )
                            st.download_button(
                                "PNG",
                                data=get_cached_png_bytes(
                                    f"dl_png_domain_cross_{grp_name}_{socio_name}_{mod_name}",
                                    fig_cross,
                                    data_signature,
                                ),
                                file_name=f"{grp_name}_{socio_name}_{mod_name}.png",
                                mime="image/png",
                                key=f"dl_png_domain_cross_{grp_name}_{socio_name}_{mod_name}",
                            )
                        st.pyplot(fig_cross, use_container_width=True)

                    rows = []
                    for c in cross_cols:
                        pct = df_cross[c].value_counts(normalize=True).mul(100).reindex(ORDER_LEVELS, fill_value=0)
                        row = {"Sous-domaine": c}
                        for lvl in ORDER_LEVELS:
                            row[lvl] = float(pct[lvl])
                        row["Total"] = float(sum(row[lvl] for lvl in ORDER_LEVELS))
                        rows.append(row)
                    cross_table = pd.DataFrame(rows)
                    st.dataframe(format_df_for_display(cross_table), use_container_width=True)

with tab4:
    st.subheader("Croisement socio-demographique selon l'outcome")
    socio_candidates = [smart_find_column(list(prepared.columns), x) for x in SOCIO_VARS]
    socio_candidates = [x for x in socio_candidates if x]

    if not socio_candidates or not outcome_cols:
        st.warning("Variables insuffisantes pour l'analyse bivariee.")
    else:
        b1, b2 = st.columns(2)
        socio_col = b1.selectbox("Covariable", socio_candidates)
        out_col = b2.selectbox("Outcome", outcome_cols)

        fig, ct = build_bivariate_figure(prepared, socio_col, out_col)
        if ct is None:
            st.warning("Pas de donnees exploitables pour ce croisement.")
        else:
            _, bcol = st.columns([8, 1])
            with bcol:
                socio_name = re.sub(r"[\\/*?:\"<>|]", "_", str(socio_col))
                out_name = re.sub(r"[\\/*?:\"<>|]", "_", str(out_col))
                st.download_button(
                    "PNG",
                    data=get_cached_png_bytes(f"dl_png_biv_{socio_name}_{out_name}", fig, data_signature),
                    file_name=f"{socio_name}_x_{out_name}.png",
                    mime="image/png",
                    key=f"dl_png_biv_{socio_name}_{out_name}",
                )
            st.pyplot(fig, use_container_width=True)
            st.dataframe(format_df_for_display(ct), use_container_width=True)

with tab5:
    st.subheader("Exports")
    base_name = re.sub(r"\.[^.]+$", "", uploaded.name)
    base_name = re.sub(r"[\\/*?:\"<>|]", "_", base_name)
    socio_candidates_export = [smart_find_column(list(prepared.columns), x) for x in SOCIO_VARS]
    socio_candidates_export = [x for x in socio_candidates_export if x]

    full_bytes = export_base_with_tests_excel(raw_df, prepared, domain_map, outcome_cols)
    st.download_button(
        "Telecharger base_et_tests.xlsx",
        data=full_bytes,
        file_name=f"{base_name}_base_et_tests.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uni_bytes = make_univariate_export(prepared)
    st.download_button(
        "Telecharger analyse_univariee.xlsx",
        data=uni_bytes,
        file_name=f"{base_name}_analyse_univariee.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    bivar_bytes = export_bivariate_excel(prepared, outcome_cols)
    st.download_button(
        "Telecharger analyse_bivariee.xlsx",
        data=bivar_bytes,
        file_name=f"{base_name}_analyse_bivariee.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    all_biv_zip = export_bivariate_graphs_zip(prepared, socio_candidates_export, outcome_cols)
    st.download_button(
        "Telecharger tous les croisements (ZIP)",
        data=all_biv_zip,
        file_name=f"{base_name}_croisements_bivaries.zip",
        mime="application/zip",
        key="dl_zip_all_biv_crossings",
    )

    auto_save_key = f"auto_saved_biv_zip|{base_name}|{data_signature}"
    if not st.session_state.get(auto_save_key):
        try:
            current_dir = Path.cwd()
            zip_path = current_dir / f"{base_name}_croisements_bivaries.zip"
            extract_dir = current_dir / f"{base_name}_croisements_bivaries"

            zip_path.write_bytes(all_biv_zip)
            with zipfile.ZipFile(io.BytesIO(all_biv_zip), mode="r") as zf:
                zf.extractall(extract_dir)

            st.session_state[auto_save_key] = str(extract_dir)
            st.success(f"Enregistrement automatique termine: {extract_dir}")
        except Exception as exc:
            st.error(f"Echec d'enregistrement local automatique: {exc}")
    else:
        st.info(f"Fichiers deja enregistres dans: {st.session_state[auto_save_key]}")

st.sidebar.markdown("---")
st.sidebar.caption("App COPSOQ: nettoyage + scoring + visualisation")




