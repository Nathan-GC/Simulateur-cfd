"""Application web Streamlit du simulateur cfd2d (point d'entrée : ``app.py`` à la racine).

Organisation :

* :mod:`webapp.common` : chemins, libellés, mise en forme, charte graphique des figures ;
* :mod:`webapp.params` : contrôles de l'interface, paramètres, configuration du solveur,
  diagnostic avant calcul et estimation du coût ;
* :mod:`webapp.preview` : schéma du domaine et aperçu en direct ;
* :mod:`webapp.runner` : exécution des calculs en arrière-plan, moniteurs, prolongation ;
* :mod:`webapp.analysis` : post-traitement et lecture commentée des résultats ;
* :mod:`webapp.figures` : graphiques interactifs (Plotly) et figures matplotlib ;
* :mod:`webapp.results` : affichage des résultats d'un calcul ;
* :mod:`webapp.store` : enregistrement des calculs sur disque, historique ;
* :mod:`webapp.history` : historique, comparaison et études paramétriques ;
* :mod:`webapp.scenarios` : scénarios pédagogiques guidés ;
* :mod:`webapp.codegen` : script Python équivalent aux réglages, extraits du code source ;
* :mod:`webapp.docs` : onglet « Théorie & Documentation ».

Les calculs eux-mêmes sont ceux du paquet ``cfd2d`` (dossier ``src/cfd2d``).
"""
