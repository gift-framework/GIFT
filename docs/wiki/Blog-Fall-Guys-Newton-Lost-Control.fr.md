---
title: "Blog : Le jour où Newton a perdu le contrôle"
layout: default
---

> Version originale en anglais sur [giftheory.substack.com](https://giftheory.substack.com/p/episode-1-the-day-newton-lost-control)

# Épisode 1 : Le jour où Newton a perdu le contrôle

*Épisode 1 : Fall Guys et le mythe de la physique prévisible.*

---

Tu connais le parcours par cœur. Tu l'as fait vingt fois en mode entraînement. Tu sais où sauter, quand glisser, où esquiver les marteaux. Sur un parcours vide, tu finis en moins de deux minutes, à chaque fois, sans réfléchir.

Tu lances une partie en multi. 60 jellybeans sur la grille de départ. Le buzzer sonne. Tu cours.

Trois mètres avant la première porte basculante, ça commence à coincer. Six jellybeans devant toi forment un tas qui ne pousse plus la porte. Tu essaies de contourner. Un septième jellybean dévale la pente derrière toi et te projette dans le tas. Tu es maintenant à plat ventre. La porte bascule sur un autre coup d'épaule à l'autre extrémité du couloir, mais tu es coincé sous le poids de quatre coéquipiers.

Quand tu te relèves, la moitié du peloton est passée. Tu sprintes vers la deuxième zone, la balançoire géante. Tu arrives en surnombre du mauvais côté. La plateforme bascule. Tu glisses dans le vide.

Éliminé en une minute trente. Sur un parcours que tu maîtrises.

Tu n'as pas mal joué. Tu n'as pas perdu. Tu as été *propagé*. Une cascade de collisions à travers une foule t'a fait traverser ce parcours d'une manière que personne n'aurait pu prédire, pas toi, pas les autres joueurs, pas le moteur du jeu lui-même tant qu'il n'avait pas calculé chaque collision.

Et voilà ce qui est intéressant : même si l'on met de côté le hasard explicite, et qu'on imagine un Fall Guys parfaitement déterministe, où chaque collision obéirait aux mêmes règles mécaniques rigides, le résultat resterait imprévisible en pratique. Newton suffirait à décrire le système. Aucune équation ne serait violée.

Et pourtant tu n'aurais rien pu prédire.

C'est l'épisode où on va déboulonner un mythe. Pas un mythe quantique, un mythe *classique*, celui qui prétend que la physique de tous les jours est intuitive et prévisible. C'est faux. Et tu le sais déjà parce que tu joues à Fall Guys.

---

## Le mythe de la physique prévisible

Voilà ce qu'on t'a appris à l'école sur la physique classique : tu lances une balle, tu connais sa vitesse initiale et son angle, tu calcules sa trajectoire. C'est de la mécanique de Newton. Très propre. Parfaitement prédictif. Tu peux savoir où la balle va atterrir avant qu'elle ait quitté ta main.

Et c'est cette image qui fait que beaucoup de gens, quand ils rencontrent la mécanique quantique, la trouvent *bizarre*. La quantique introduit de la probabilité, de l'incertitude, des résultats qu'on ne peut prévoir qu'en distribution. Comparée à la balle de Newton qui suit sa parabole, c'est étrange. Du coup la quantique se gagne sa réputation de *physique exotique*, comme si l'imprévisibilité était son trait distinctif.

Cette comparaison est tordue à la base. Parce que la physique classique, dès qu'on la sort du cas d'école d'une seule balle dans le vide, devient elle-même imprévisible. Pas un peu imprévisible. **Fondamentalement imprévisible**.

Le premier qui l'a démontré rigoureusement s'appelle Henri Poincaré. Fin du XIXe siècle, il bosse sur le problème à trois corps. Trois objets célestes en interaction gravitationnelle mutuelle, Soleil, Terre, Lune par exemple. Tu connais leur position et leur vitesse à un instant donné. Question : où seront-ils dans dix ans ?

La physique de Newton dit que la réponse existe. Les équations sont déterministes. En principe, tu calcules.

Poincaré découvre que **tu ne peux pas calculer**. Pas parce que les ordinateurs étaient trop faibles à son époque (ils étaient inexistants), parce que le problème lui-même est *intrinsèquement* sensible. Si tu connais les positions à un milliardième de mètre près, ton erreur sur la position dans dix ans est de plusieurs millions de kilomètres. Si tu connais à un milliardième de milliardième de mètre près, ton erreur sur cent ans est aussi grande. Aucune précision finie ne suffit. C'est ce qu'on appellera plus tard la **sensibilité aux conditions initiales**, et ce sera un des piliers de la théorie du chaos au XXe siècle.

Pierre-Simon de Laplace, un siècle plus tôt, avait imaginé un démon hypothétique qui, connaissant à un instant la position et la vitesse de toutes les particules de l'univers, pourrait calculer le passé et le futur entiers. Cette image était l'idéal newtonien : déterminisme parfait égale prédictibilité parfaite.

Poincaré ne détruit pas le déterminisme mathématique. Il détruit quelque chose de plus concret : l'idée qu'une connaissance *finie*, même extraordinairement précise, suffise à prédire durablement un système réel. Les deux idées sont distinctes. Tu peux avoir un univers totalement déterministe dans ses équations et totalement imprévisible dans ses prédictions opérationnelles.

Le mot a fait son chemin dans la culture pop avec le *butterfly effect* d'Edward Lorenz : un battement d'ailes de papillon au Brésil peut provoquer une tornade au Texas. Ce n'est pas une métaphore poétique, c'est un fait technique sur les équations de la météo. Et si tu as déjà regardé une prédiction météo à dix jours déraper complètement, tu as vécu Poincaré sans le savoir.

Mais il y a un domaine où tu vis Poincaré tous les soirs sans en avoir conscience. C'est Fall Guys.

---

## Pourquoi Fall Guys est un laboratoire de physique des foules

Voilà ce que les concepteurs de Fall Guys ont fait, qu'ils l'aient prémédité ou non : ils ont construit, sans forcément le chercher, une petite simulation jouable d'un système physique qui occupe un des champs de recherche les plus actifs de la mécanique contemporaine. Le système s'appelle un **milieu granulaire**, et son cousin humain s'appelle la **physique des foules**.

Un milieu granulaire, c'est un ensemble de petits objets indépendants, grains de sable, billes, jellybeans, qui interagissent par contact mécanique. Frottements, collisions, gravité. Pas de mystère individuellement. Mais quand tu en mets beaucoup ensemble, des comportements collectifs émergent qu'aucune particule ne porte seule.

Le sable dans un sablier est un exemple. Quand le passage est large, le sable coule comme un fluide. Quand le passage est étroit, des **arches mécaniques** se forment spontanément où plusieurs grains se bloquent mutuellement, créant une voûte qui supporte tout le poids au-dessus. Le sable se transforme alors en *solide* le temps de quelques secondes. Puis l'arche s'effondre, le sable reprend son écoulement, jusqu'à la prochaine arche. C'est une **transition de phase intermittente** entre deux états de matière, sans changement de température ni de composition. Juste de la géométrie.

Dans Fall Guys, tu vis ça à chaque parcours qui implique un goulot d'étranglement.

### Door Dash : la foule comme matière granulaire

*Door Dash*, où il faut deviner les vraies portes parmi les fausses : le tas de jellybeans devant chaque porte n'est pas un agglomérat passif. C'est une foule en transition de phase. Tant que la densité reste fluide, ça avance. Au-dessus d'une densité critique, ça se solidifie, et personne n'avance plus, même pas ceux qui ont la bonne porte. Tu attends que quelqu'un se fasse pousser hors du tas pour que le flux reprenne. Comme une arche dans le sablier qui finit par céder.

Cette dynamique a été modélisée par des physiciens depuis les années 90. Dirk Helbing, à Zurich, a publié une série de papiers fondateurs sur la *physique des foules*. Sa découverte la plus contre-intuitive : si tu mets un *obstacle* devant la sortie d'une salle d'évacuation, les gens sortent **plus vite**. Parce que l'obstacle empêche la formation d'arches devant la porte. Le tas se restructure, le flux redevient fluide. Les architectes utilisent cette idée pour concevoir les sorties de stade. Et tu l'as vécue, sans le savoir, chaque fois que tu as réussi à passer une porte basculante quand tout le monde est resté coincé.

### See Saw : bifurcation catastrophique

Le **See Saw**, ces longues balançoires bleues, c'est un autre régime. Ce n'est plus de la physique granulaire, c'est un système avec **rétroaction positive et bifurcation catastrophique**. La plateforme penche en fonction du poids cumulé d'un côté. Plus de jellybeans donc plus de pente, donc ils glissent plus vite vers le côté lourd, donc la pente s'accentue, donc ils sont projetés. C'est une **boucle d'amplification**, exactement comme certaines transitions de phase de premier ordre en physique. Un petit déséquilibre se mange lui-même jusqu'à la catastrophe. Tu peux essayer de "tenir" en haut. Sauf que si trois autres jellybeans ont la même idée et arrivent en même temps que toi, le poids cumulé bascule la plateforme, et vous tombez tous ensemble. Tu n'as rien fait de mal. Tu as juste été un grain de plus dans le moment de bifurcation.

### Big Yeetus : perturbation externe d'un système ouvert

Le **Big Yeetus**, ce gros marteau orange qui frappe au hasard certains parcours, c'est encore autre chose. C'est une **perturbation extérieure non corrélée** au reste du système. Sa fonction n'est pas de tuer les joueurs, c'est de **redistribuer les états**. Avant Big Yeetus, le peloton se distribuait en tête-milieu-queue selon les compétences. Après Big Yeetus, six jellybeans qui étaient en tête se retrouvent en queue, et trois qui étaient en queue se retrouvent en tête. La distribution est recalculée. Tu ne peux pas t'y préparer parce que c'est par construction asynchrone avec ta stratégie.

Ce n'est pas encore de la quantique, évidemment. Mais c'est une bonne image d'un *système ouvert* : un système qui n'évolue pas en vase clos, mais qui reçoit régulièrement des injections d'énergie venues de l'extérieur, redistribuant ses trajectoires sans logique interne au système. La plupart des systèmes physiques réels sont ouverts en ce sens. Et Big Yeetus te le rappelle douloureusement.

### Hex-A-Gone : agir et observer ne sont pas séparables

Et puis il y a *Hex-A-Gone*. Les hexagones qui disparaissent au contact du joueur. C'est le seul mode où tu détruis activement ton environnement par ta seule présence. Tu marches sur un hexagone, il s'efface. Tu sautes sur le suivant, il s'efface. Plus tu joues longtemps, moins il reste de plateforme. La survie consiste à *minimiser ton interaction* avec l'environnement, sauter, anticiper, se tenir aux bords. Hex-A-Gone illustre quelque chose qu'on retrouvera, mais pas tout à fait, en quantique : **agir et observer ne sont pas séparables**. Sauf qu'en quantique c'est plus subtil que ça, on raffinera l'image dans plusieurs épisodes.

---

## Le moment où le chaos disparaît

Voilà la chose la plus belle dans Fall Guys, et la plus instructive : **le chaos disparaît en finale**.

Quand il ne reste que cinq jellybeans en finale, le comportement change radicalement. Tu peux à nouveau prévoir. Tu peux à nouveau jouer *stratégiquement* au lieu de subir des cascades. Le parcours redevient ce qu'il était en mode entraînement : un défi technique avec des règles compréhensibles.

Pourquoi ? Parce que la **densité a chuté**. La physique granulaire et la physique des foules ne sont pas des phénomènes d'objets, ce sont des phénomènes de *concentration d'objets*. Le même jellybean qui était imprédictible à 60 redevient prédictible à 5. Sa physique individuelle n'a pas changé. Ce qui a changé, c'est le **régime collectif** dans lequel il évolue.

C'est exactement la transition entre régime turbulent et régime laminaire en mécanique des fluides. À haut nombre de Reynolds (= flux dense, énergique, complexe), l'eau d'une rivière forme des tourbillons, des cascades, des structures émergentes que personne ne peut prédire individuellement. À bas nombre de Reynolds (= flux lent, dilué), la même eau coule en couches parallèles bien rangées, parfaitement prédictibles. Mêmes équations, mêmes molécules d'eau. Mais deux mondes physiques différents.

Ta finale Fall Guys, c'est le régime laminaire. Tes premiers parcours de qualification, c'est le régime turbulent. Les concepteurs du jeu te font littéralement traverser une **transition de phase** entre les deux. Ils ne te l'ont jamais dit. Tu l'as appris par les fesses (littéralement, à chaque chute).

---

## Ce que tu sais maintenant que tu ne savais pas avant

Si tu as suivi jusqu'ici, tu viens de comprendre quelque chose que même beaucoup de physiciens passent à côté : **la physique classique n'est pas ce royaume tranquille et parfaitement prévisible qu'on oppose trop souvent à la physique quantique**. Elle possède déjà ses propres formes d'imprévisibilité : chaos, turbulence, foules, transitions collectives. La quantique n'arrive donc pas dans un monde classique simple ; elle ajoute un autre régime d'étrangeté à une physique qui était déjà non triviale.

Cette découverte change la nature de la question quantique. Avant Fall Guys, tu pouvais croire que l'imprévisibilité quantique était l'exception bizarre dans un univers par ailleurs Newtonien et tranquille. Après Fall Guys, tu sais que l'imprévisibilité est *partout*. Dans les particules subatomiques. Dans le climat. Dans une foule à la sortie d'un stade. Dans 60 jellybeans devant une porte basculante. Ce qui change selon les régimes, c'est *de quelle manière* le système devient imprévisible.

La quantique a sa manière propre, qui rend les choses imprévisibles à très petite échelle pour des raisons spécifiques (interférences, intrication, principe d'incertitude). On y reviendra. Et il faut être précis : le chaos classique et l'indétermination quantique ne sont pas le même type d'imprévisibilité. Le premier est une limite *pratique* : les variables existent, mais aucune mesure finie ne suffit à les capturer. Le second est, dans l'interprétation standard, une indétermination *intrinsèque* : certaines valeurs ne peuvent pas être considérées comme déjà fixées avant la mesure. C'est ce que Bell a montré dans l'épisode précédent : pour un certain type de mesure quantique, aucune valeur préexistante ne peut rendre compte des corrélations observées.

Mais ce qu'il faut retenir, c'est ceci : la quantique n'est plus une exception isolée dans une physique sereine. Elle est un **régime parmi d'autres** dans une physique qui, prise dans son ensemble, est partout traversée par le chaos et l'émergence.

Ça veut dire quelque chose de plus profond pour la suite de cette série. Quand on va parler de la mesure quantique, de la non-commutativité, de l'intrication, tu n'auras pas à les accueillir comme des phénomènes *bizarres* qui violent une intuition classique tranquille. Tu pourras les accueillir comme des **régimes plus fins** d'imprévisibilité, qui s'ajoutent au régime grossier de la mécanique des foules.

Tu n'es plus dans la position de défendre une physique commode contre des intrus quantiques. Tu es dans la position de quelqu'un qui sait que **toute physique réelle est non-triviale**, et qui peut maintenant explorer les différents visages de cette non-trivialité avec curiosité au lieu de méfiance.

C'était la promesse de cette série. On commence à la tenir.

---

*Cette série explore une douzaine de jeux et leurs concepts physiques associés. Pour ne rien rater, abonne-toi sur [giftheory.substack.com](https://giftheory.substack.com/).*

🌀
