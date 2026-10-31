# Was sich inhaltlich geändert hat – ausführlich erklärt

Diese Datei erklärt die inhaltlichen Unterschiede zwischen dem alten Code (git `HEAD` vor dem Umschreiben) und
dem neuen Paket `finalfits/`: was vorher anders oder falsch war, warum, was es physikalisch bedeutet und was
jetzt passiert. Wie der Code geschrieben ist (Paketstruktur, Logging, Tests), spielt hier keine Rolle. Die
knappe englische Liste steht in [DIFFERENCES.md](DIFFERENCES.md).

Alle Zahlen stammen aus deinen echten 2023-NJ-Ausgaben (`output_2023_NJ_new_bkg/`, 2023preBPix, mH = 125 GeV),
soweit nicht anders angegeben. Für die neuen Werte habe ich dieselben HiggsDNA-Eingaben mit dem neuen Code
verarbeitet.

---

## Inhalt

1. [Überblick: alle Unterschiede auf einen Blick](#1-überblick-alle-unterschiede-auf-einen-blick)
2. [Trees2WS: die Systematik-Histogramme (der wichtigste Fehler)](#2-trees2ws-die-systematik-histogramme-der-wichtigste-fehler)
3. [Datacard: Bereinigen und Symmetrisieren der lnN-Werte](#3-datacard-bereinigen-und-symmetrisieren-der-lnn-werte)
4. [Combine](#4-combine)
5. [Plots](#5-plots)
6. [Unterschiede ohne Einfluss auf Ergebnisse](#6-unterschiede-ohne-einfluss-auf-ergebnisse)
7. [Was du neu laufen lassen musst](#7-was-du-neu-laufen-lassen-musst)
8. [Offene Punkte, die du entscheiden solltest](#8-offene-punkte-die-du-entscheiden-solltest)

---

## 1. Überblick: alle Unterschiede auf einen Blick

| # | Schritt | Was war vorher | Ändert es deine Ergebnisse? |
|---|---|---|---|
| 2 | Trees2WS | ~90 % der Systematik-Histogramme leer, der Rest teils mit falschen Events | **Ja, stark:** Signalmodell und Datacard |
| 3a | Datacard-Cleaner | Faktor-Grenze ausgeschaltet, sobald `symmetrizeNuisance` gesetzt ist | Bei deiner Config nein (siehe 3d) |
| 3b | Datacard-Cleaner | Symmetrisierung verliert die Unsicherheit oder dreht ihre Richtung um | Bei deiner Config nein (siehe 3d) |
| 3c | Datacard-Cleaner | Asymmetrische `lumi`-Werte doppelt geschrieben → Datacard kaputt | Nein, deine `lumi`-Werte sind symmetrisch |
| 4a | Combine | Korrelationsmatrix immer mit 2022-Luminosität beschriftet | Nur die Beschriftung |
| 4b | Combine | Post-Fit-Toys stürzen außerhalb von slurm immer ab | Post-Fit-Bänder jetzt auch auf htcondor möglich |
| 4c | Combine | `UnblindedFitSystSingle` auf PSI kopiert falsch | Nur auf PSI |
| 4d | Combine | pdf-Indizes: bei fehlender Datei Müll in `--setParameters` | Nur im Fehlerfall |
| 4e | Combine | `PrepareTheDirectory` stürzt bei getrennten Modell-Ordnern ab | Nur in dieser Konstellation |
| 4f | Combine | p-Wert: irreführender `TypeError` | Nur im Fehlerfall |
| 5a | Spektren | Theorie-Boxen von MadGraph im Ratio-Panel oben/unten vertauscht | **Ja, sichtbar im Plot** |
| 5b | S+B-Plots | Problematische Kategorien in Toy-Bändern mitgezählt; mit Gewichten Absturz | Nur mit `--problematicCats` |
| 5c | Impacts | `groups` als ungültiges JSON | Nein, deine `groups` sind leer |
| 5d | Korrelationsmatrix | Absturz bei POI ohne Übersetzung | Nur im Fehlerfall |
| 5e | S+B-Plots | Fehler mit Exit-Code 0 (law dachte „erfolgreich“) | Nur im Fehlerfall |
| 6 | verschiedene | Ablageort der fTest-Datei, Reihenfolge der Jahre, Startwert von `CMS_hgg_mass` | Nein |

Alles, was hier nicht steht, liefert **bitgenau dieselben** Ergebnisse wie vorher: Background-fTest und Envelopes,
Signal-fTest, Signal-Fit, Packaging, Yields, Datacard, text2workspace und alle 284 278 combine-Befehle.

---

## 2. Trees2WS: die Systematik-Histogramme (der wichtigste Fehler)

### 2.1 Wofür diese Histogramme gebraucht werden

HiggsDNA schreibt für jede experimentelle Systematik (Jet-Energieskala, Photon-Energieskala, Smearing, …) einen
eigenen Baum mit den Events, deren Energie um ±1σ verschoben wurde. Trees2WS macht daraus pro Prozess,
Kategorie und Variation ein `RooDataHist`, z. B.

```
vbf_125_13TeV_RECO_NJ_1p0_2p0_cat0                         ← nominal
vbf_125_13TeV_RECO_NJ_1p0_2p0_cat0_JecSystTotalUp01sigma   ← Jet-Energieskala +1σ
vbf_125_13TeV_RECO_NJ_1p0_2p0_cat0_ScaleEBZeeDown01sigma   ← Photon-Energieskala Barrel −1σ
...
```

Diese Histogramme gehen an zwei Stellen in die Analyse ein:

```
                    ┌──────────────────────────────┐
HiggsDNA-Bäume ───► │ Trees2WS: Systematik-Hists   │
                    └──────────────┬───────────────┘
              ┌────────────────────┴─────────────────────┐
              ▼                                          ▼
  CalcPhotonSyst (Signal)                     MakeYields / Datacard
  Wie stark verschieben Photon-Skala          Wie stark ändert die Jet-Energie-
  und Smearing Peak und Breite des            skala (JEC/JER) und SigmaEOverE
  Signals? → param-Nuisances im               die Anzahl der Events? → lnN-
  Signalmodell                                Nuisances in der Datacard
```

### 2.2 Was der alte Code falsch gemacht hat

Für die differentielle Aufteilung (z. B. NJ-Bins) muss jedes Event einem Generator-Bin zugeordnet werden. Der
alte Code hat dafür eine Auswahl-Maske auf dem **nominalen** Baum berechnet und dieselbe Maske dann auf die
**Systematik-Bäume** angewendet. Die Systematik-Bäume enthalten aber andere Events in anderer Reihenfolge
(Events wandern durch die Energieverschiebung in andere Kategorien oder fallen aus der Selektion). Die Maske
passte also nicht zu den Events.

Ein Spielzeug-Beispiel mit fünf Events:

```
Nominaler Baum:        Event  A    B    C    D    E
Generator-Bin:                0j   1j   1j   2j   0j
Maske "1-Jet-Bin":            ✗    ✓    ✓    ✗    ✗     → Position 2 und 3

Systematik-Baum:       Event  B    D    E    F         (andere Events, andere Reihenfolge, andere Länge)
alte Maske angewendet:        ✗    ✓    ✓    ?         → D (2-Jet!) und E (0-Jet!) landen im 1-Jet-Bin
richtig wäre:                 ✓    ✗    ✗    …         → nur B gehört in den 1-Jet-Bin
```

Je nach Länge der Bäume ergab das entweder **leere** Histogramme oder Histogramme mit **falschen Events**.

### 2.3 Wie schlimm war es? Echte Zahlen

Alle VBF-Workspaces von 2023preBPix bei mH = 125 GeV, je 2400 Systematik-Histogramme:

| | in-fiducial | out-of-fiducial |
|---|---|---|
| Systematik-Histogramme | 1200 | 1200 |
| **alt: leer** | **1075 (90 %)** | 1104 (92 %) |
| alt: gefüllt, aber > 5 % neben dem richtigen Wert | 69 | – |
| **neu: leer** | **0** | 1200 (siehe 2.6) |

Ein Beispiel, Kategorie `RECO_NJ_1p0_2p0_cat0`, Prozess `vbf_NJ_1p0_2p0_in` (Summe der Gewichte):

| Histogramm | alt | neu |
|---|---|---|
| nominal | 0.0241 | 0.0241 |
| JecSystTotal Up / Down | **0 / 0** | 0.0241 / 0.0239 |
| JerSyst Up / Down | **0 / 0** | 0.0239 / 0.0243 |
| ScaleEBZee Up / Down | **0 / 0** | 0.0241 / 0.0241 |
| Smearing Up / Down | **0 / 0** | 0.0241 / 0.0241 |
| energyErrShift Up / Down | **0 / 0** | 0.0214 / 0.0226 |

Und ein Beispiel für die „falsch gefüllten“: ein VBF-Prozess in `RECO_NJ_0p0_1p0_cat0` mit nominal 0.0018 hatte
alt bei JEC-Down **0.0115** (das 6-fache des nominalen Werts) statt richtig 0.0021.

### 2.4 Folge 1: Photon-Systematiken des Signalmodells standen auf ihren Obergrenzen

`CalcPhotonSyst` vergleicht Mittelwert, Breite und Rate des Signalpeaks zwischen nominal und ±1σ. Die
Ergebnisse werden auf Obergrenzen begrenzt: 5 % für den Mittelwert, 50 % für die Breite, 5 % für die Rate.
Ein leeres Histogramm hat Mittelwert 0 und Breite 0, also eine Abweichung von 100 %, und landet damit genau auf
der Obergrenze.

In den alten Photon-Systematiken (alle Prozesse und Kategorien, 2023preBPix, in-fiducial, 300 Einträge):

| Größe | Einträge auf der Obergrenze |
|---|---|
| ScaleEBZee Mittelwert (Grenze 5 %) | 255 von 300 |
| ScaleEBZee Breite (Grenze 50 %) | 260 von 300 |
| Smearing Breite (Grenze 50 %) | 263 von 300 |

Direkter Vergleich alt/neu für `RECO_NJ_2p0_3p0_cat0` (VBF, in-fiducial):

| Prozess | Mittelwert-Shift ScaleEBZee alt → neu | Breite ScaleEBZee alt → neu | Breite Smearing alt → neu |
|---|---|---|---|
| vbf_NJ_0p0_1p0_in | 5 % → 0.042 % | 50 % → 13.9 % | 50 % → 12.7 % |
| vbf_NJ_1p0_2p0_in | 5 % → 0.073 % | 50 % → 1.6 % | 50 % → 2.9 % |
| **vbf_NJ_2p0_3p0_in** (diagonal) | **5 % → 0.067 %** | **50 % → 2.4 %** | **50 % → 3.2 %** |
| vbf_NJ_3p0_4p0_in | 5 % → 0.077 % | 50 % → 3.1 % | 50 % → 4.5 % |
| vbf_NJ_4p0_100p0_in | 5 % → 1.05 % | 50 % → 50 % | 50 % → 16.6 % |

**Was das physikalisch bedeutet:** Die Werte werden im Signalmodell zu `param`-Nuisances. Ein Mittelwert-Shift von
5 % pro 1σ heißt, dass der Fit den Higgs-Peak pro Standardabweichung der Photon-Energieskala um
5 % × 125 GeV ≈ **6.3 GeV** verschieben durfte. Richtig sind etwa 0.07 % ≈ **0.08 GeV**. Genauso durfte die Peak-Breite
pro 1σ um 50 % statt um wenige Prozent schwanken.

```
Signalform bei Nuisance θ = +1σ (schematisch, Diphoton-Masse in GeV)

alt:    nominal ▁▂▅█▅▂▁                     verschoben ▁▁▂▃▅▃▂▁▁       ← 6 GeV weiter, 50 % breiter
             120  125  130                         130  131  136

neu:    nominal ▁▂▅█▅▂▁   verschoben ▁▂▅█▅▂▁                          ← praktisch deckungsgleich
             120  125  130          120  125  130
```

Das Signalmodell hatte dadurch eine unrealistische Freiheit in Position und Breite. Ich erwarte, dass die
Unsicherheiten der Signalstärken (auch in den Asimov-Fits) dadurch zu groß waren. Wie groß der Effekt ist,
zeigt erst ein neuer Lauf, das habe ich nicht gemessen.

### 2.5 Folge 2: In der Datacard fehlten JEC, JER und SigmaEOverE fast überall

Die Datacard berechnet für Histogramm-Systematiken den lnN-Wert als Verhältnis von Up- und Down-Ausbeute zu
ihrem Mittelpunkt. Ist eine der Ausbeuten 0, wird der Wert auf „keine Wirkung“ (`-`) gesetzt. In deiner alten
Datacard (`Datacard_NJ_2023_unsymmetrized.txt`) heißt das:

| Nuisance | Signalspalten 2023preBPix mit Wert |
|---|---|
| `CMS_scale_j_2023preBPix` (Jet-Energieskala) | **7 von 230** |
| `CMS_HIG26007_SigmaEOverEShift` | **7 von 230** |
| zum Vergleich `CMS_pileup` (Gewicht, nicht betroffen) | 193 von 230 |

Diese Unsicherheiten fehlten also fast vollständig. Gerade für eine NJ-Messung ist die Jet-Energieskala eine
wichtige Unsicherheit, weil sie Events zwischen den Jet-Bins verschiebt. Die erwarteten Unsicherheiten waren in
diesem Punkt also zu **klein**, während das Signalmodell (2.4) zu viel Freiheit hatte. Beide Fehler wirken in
entgegengesetzte Richtungen und heben sich nicht sauber auf.

### 2.6 Was der neue Code macht – und was er nicht kann

Der neue Code ordnet die Events der Systematik-Bäume über den HiggsDNA-Bin-Index (`diffVar`) zu. Dieser Index
steht in jedem Baum am Event selbst, also gibt es kein Reihenfolge-Problem mehr. Er wird nur verwendet, wo er
auf den nominalen Events denselben Bin ergibt wie die Generator-Variable. Für alle in-fiducial-Bins ist das der
Fall: **0 von 1200 Histogrammen sind jetzt leer.**

**Einschränkung out-of-fiducial:** HiggsDNA teilt die Events außerhalb des fiduziellen Volumens nicht nach Bins
auf, und die Systematik-Bäume enthalten keine Generator-Variable (z. B. `GenNJ`). Diese Events kann man deshalb
keinem Bin zuordnen. Ihre Histogramme bleiben leer, und der neue Code warnt.

**Wichtig:** Das hat dieselben Folgen wie vorher, nur für die out-of-fiducial-Prozesse. Ihre Photon-Systematiken
stehen weiterhin auf den Obergrenzen, z. B. `vbf_NJ_2p0_3p0_out` in `RECO_NJ_2p0_3p0_cat0`: Mittelwert 5 %,
Breite 50 %. Ihre JEC-Werte in der Datacard bleiben `-`. Die Ausbeuten dieser Prozesse sind klein (in derselben
Kategorie 0.00013 gegenüber 0.032 für `vbf_NJ_2p0_3p0_in`, also etwa 0.4 %), der Effekt auf die Fits ist also viel kleiner als vorher, aber
nicht null. Wie man damit umgeht, steht in [Abschnitt 8](#8-offene-punkte-die-du-entscheiden-solltest).

---

## 3. Datacard: Bereinigen und Symmetrisieren der lnN-Werte

### 3.0 Kurz zur Konvention

Ein lnN-Eintrag beschreibt, wie sich eine Ausbeute N mit der Nuisance θ ändert:

- symmetrisch `κ`: N(θ) = N · κ^θ, also bei θ = +1 ⇒ N · κ und bei θ = −1 ⇒ N / κ
- asymmetrisch `κ_down/κ_up`: bei θ = −1 ⇒ N · κ_down, bei θ = +1 ⇒ N · κ_up

Beispiel: `0.950/1.050` heißt −5 % bei −1σ und +5 % bei +1σ, gleichwertig mit symmetrisch `1.050`.

Der Cleaner (`cleanDatacard.py` alt, `scripts/datacard.py clean` neu) entfernt unsinnige Werte (außerhalb von
`[1/factor, factor]`) und kann für ausgewählte Nuisances einseitige Werte symmetrisieren (`symmetrizeNuisance`
in der Config).

### 3a. Die Faktor-Grenze war ausgeschaltet, sobald symmetrisiert wurde

Im alten Code waren die Faktor-Prüfung und die Symmetrisierung zwei Zweige eines `if/else`. War
`symmetrizeNuisance` gesetzt (bei dir ist es das), lief die Faktor-Prüfung für **keine** asymmetrische
Nuisance mehr, auch nicht für die, die gar nicht symmetrisiert werden sollen.

| Eintrag (factor = 10000) | ohne Symmetrisierungsliste | mit Liste, alt | mit Liste, neu |
|---|---|---|---|
| `0.00005/1.050` | `1.000/1.050` | **`0.000/1.050`** | `1.000/1.050` |

`κ_down = 0.00005` heißt, dass der Prozess bei −1σ praktisch verschwindet. So etwas ist ein numerisches Artefakt
und genau das, was der Faktor entfernen soll. Neu gilt die Grenze immer.

### 3b. Die Symmetrisierung hat Unsicherheiten verloren oder umgedreht

Symmetrisiert werden sollen einseitige Einträge: Eine Seite liegt bei 1 (Abstand < 0.003), die andere weicht ab
(Unterschied ≥ 0.010). Der alte Code hatte dafür eine Bedingung, die nie erfüllt sein kann
(`low < high and low − high ≥ 0.010`). Deshalb landeten drei von vier Fällen im falschen Zweig:

| Eintrag | Bedeutung | alt | Wirkung alt | neu | Wirkung neu |
|---|---|---|---|---|---|
| `1.000/1.050` | +5 % bei +1σ | `1.000` | **Unsicherheit weg** | `1.050` | +5 % bei +1σ ✓ |
| `0.950/1.000` | −5 % bei −1σ | `0.950` | −5 % bei **+1σ**: **Richtung umgedreht** | `1.053` | −5 % bei −1σ ✓ |
| `1.050/1.000` | +5 % bei −1σ | `1.050` | +5 % bei **+1σ**: **Richtung umgedreht** | `0.952` | +5 % bei −1σ ✓ |
| `1.000/0.950` | −5 % bei +1σ | `0.950` | ✓ | `0.950` | ✓ |

Neu gilt: Weicht die Up-Seite ab, wird `κ = κ_up`. Weicht die Down-Seite ab, wird `κ = 1/κ_down`. So bleibt die
Seite, auf der etwas passiert, mit Betrag und Richtung erhalten. Die Richtung ist wichtig, sobald eine Nuisance
mehrere Prozesse oder Kategorien gleichzeitig beeinflusst (Korrelationen, Migrationen zwischen Bins).

### 3c. Asymmetrische `lumi`-Werte wurden doppelt geschrieben

Für Nuisances mit „lumi“ im Namen hängte der alte Code den Wert an und lief dann **zusätzlich** in den normalen
Zweig, der ihn noch einmal anhängte:

```
Eingabe:  lumi_X  lnN   0.980/1.020   -       1.010
alt:      lumi_X  lnN   0.980/1.020   0.980/1.020   -   1.010      ← eine Spalte zu viel, alles verrutscht
neu:      lumi_X  lnN   0.980/1.020   -       1.010
```

Deine `lumi`-Werte sind alle symmetrisch, deshalb war deine Datacard davon nicht betroffen.

### 3d. Warum sich an deiner Datacard trotzdem nichts ändert

Ich habe den neuen Cleaner auf deine echte `Datacard_NJ_2023_unsymmetrized.txt` angewendet: **0 geänderte
Einträge.** Der Grund ist, dass die Namen in `symmetrizeNuisance` (`config/2023_NJ.yml`) in der Datacard gar
nicht vorkommen:

| in der Config | in der Datacard |
|---|---|
| `CMS_hgg_SigmaEOverEShift` | `CMS_HIG26007_SigmaEOverEShift` (lnN, zweiseitig) |
| `CMS_hgg_scale_0_shape`, `_1_`, `_3_` | `CMS_HIG26007_scale_0`, `_1`, `_3` |

Es wurde also weder alt noch neu irgendetwas symmetrisiert. Der neue Code warnt jetzt in diesem Fall
(„Nuisances to symmetrise without lnN line …“). Ob die Liste angepasst werden soll, ist deine Entscheidung
(siehe Abschnitt 8).

---

## 4. Combine

### 4a. Korrelationsmatrizen hatten immer die Luminosität von 2022

`Plots/makeCorrMatrix.py` schreibt die integrierte Luminosität oben rechts in den Plot und nimmt dafür `--year`
(Standard 2022). `law_combine.py` hat `--year` nie übergeben:

| Jahr der Analyse | Beschriftung alt | Beschriftung neu |
|---|---|---|
| 2022 | 34.7 fb⁻¹ | 34.7 fb⁻¹ |
| **2023** | **34.7 fb⁻¹** | **27.6 fb⁻¹** |
| 2024 | 34.7 fb⁻¹ | 109.8 fb⁻¹ |
| 2022_2023_2024 | 34.7 fb⁻¹ | 172.1 fb⁻¹ (Summe) |

Die Zahlen in der Matrix waren immer richtig, nur die Beschriftung nicht.

### 4b. Post-Fit-Toys stürzten außerhalb von slurm immer ab

`MggToyGeneration --is-postfit True` setzte am Anfang immer
`TARGET_PATH = /scratch/$USER/$SLURM_JOB_ID`. Auf htcondor oder lokal gibt es `SLURM_JOB_ID` nicht, also brach
**jeder** Post-Fit-Toy-Job mit `KeyError: 'SLURM_JOB_ID'` ab. Die ±1σ/±2σ-Bänder der Post-Fit-S+B-Plots ließen
sich damit nur auf dem PSI-Tier-3 erzeugen. Neu wird das Scratch-Verzeichnis nur auf slurm/psi benutzt, genau wie
bei den Pre-Fit-Toys.

### 4c. `UnblindedFitSystSingle` hat auf PSI anders kopiert als alle anderen Tasks

Alle Tasks legen auf slurm/psi zuerst das Zielverzeichnis an und kopieren am Ende passend zurück: mit `cp`, wenn
die Ausgabe auf `/work` liegt, sonst mit `xrdcp` auf dCache. `UnblindedFitSystSingle` hat das Anlegen
übersprungen und **immer** `xrdcp` benutzt:

```
Ausgabe auf /work:   alt: xrdcp → root://t3dcachedb03.psi.ch//work/...   ✗ (/work liegt nicht auf dCache)
                     neu: cp -rf → /work/...                              ✓
```

Liegt die Ausgabe auf `/work`, ging das Ergebnis des Fits also verloren. Neu verhält sich der Task wie alle
anderen.

### 4d. pdf-Indizes aus dem Asimov-First-Step

Für die Scans werden die im First-Step gewählten Background-Funktionen (`pdfindex_...`) per `--setParameters`
festgehalten. Sie werden mit dem Makro `checkPdfIdx.C` gelesen, das seine Werte hinter ein `X` schreibt:

```
Processing /…/Combine/checkPdfIdx.C("/…/higgsCombinefirstStep_r_NJ_0p0_1p0.MultiDimFit.mH125.07.root")...
Xpdfindex_RECO_NJ_0p0_1p0_cat0=2,pdfindex_RECO_NJ_0p0_1p0_cat1=0,...
```

Der alte Code nahm alles hinter dem **ersten** `X` in der ganzen Ausgabe. Fehlte die Datei, gab es keine
`X`-Zeile, und die ganze `Processing …`-Zeile landete in `--setParameters`. combine bekam also einen unsinnigen
Parameter. Ebenso hätte ein großes `X` im Pfad (z. B. `FidXS`) die Zerlegung zerstört. Neu wird nur die mit `X`
markierte Zeile genommen. Fehlt sie, werden keine Indizes gesetzt und es gibt eine Fehlermeldung. Wenn alles da
ist, ist das Ergebnis identisch.

### 4e. `PrepareTheDirectory` mit getrennten Ordnern

Liegen Signal- und Background-Modelle in verschiedenen Ordnern, benutzte `output()` eine Variable, die nur im
anderen Fall definiert war, und brach mit `NameError` ab. Neu funktionieren beide Fälle, bei gemeinsamem Ordner
sind die Ausgaben gleich.

### 4f. p-Wert (`PValueCalculation`)

Der p-Wert misst die Verträglichkeit mit dem Standardmodell: Ein Fit mit allen POIs fest auf 1 wird mit dem
besten Fit verglichen.

```
p = 1 − CDF_χ²(n_POI) (2 · ΔNLL)
Beispiel NJ: n_POI = 5, 2·ΔNLL = 4.6   ⇒   p = 0.467   ⇒   gerundet auf 2 signifikante Stellen: 0.47
```

Formel, Rundung und Ausgabe `pvalue.txt` sind gleich geblieben. Der Unterschied betrifft nur den Fehlerfall:

```
alt:  Datei fehlt → Lesefehler abgefangen → pvalue = None
      → round(None, …) → TypeError: unsupported format string passed to NoneType.__format__
      (die eigentlich vorgesehene Prüfung "if pvalue is not None" kam erst danach und wurde nie erreicht)

neu:  Datei fehlt                   → FileNotFoundError: … the fit at the SM point did not run or failed
      Datei ohne Eintrag am SM-Punkt → RuntimeError: … has no fixed point entry, the fit at the SM point failed
```

Ein Fehler beim Schreiben von `pvalue.txt` wird nicht mehr nur ausgegeben, sondern lässt den Task scheitern.

---

## 5. Plots

### 5a. Spektren: Theorie-Boxen von MadGraph im Ratio-Panel vertauscht

Im unteren Panel (Verhältnis zur Vorhersage) wird für jede Vorhersage eine schraffierte Box für ihre
Theorie-Unsicherheit gezeichnet. Bei POWHEG und MadGraph ohne NNLOPS waren Up und Down richtig zugeordnet, bei
**MadGraph + NNLOPS** vertauscht. Die Box ging also von `Wert − Up` bis `Wert + Down` statt von `Wert − Down`
bis `Wert + Up`.

Beispiel: Ratio 1.00, Theorie-Unsicherheit +8 % / −5 %:

```
            alt (falsch)        neu (richtig)
  1.08                          ┌───┐
  1.05      ┌───┐               │///│
  1.00  ────│///│────       ────│///│────
  0.95      │///│               └───┘
  0.92      └───┘
```

Die Höhe der Box blieb gleich, sie war nur nach unten statt nach oben verschoben. Bei symmetrischen
Unsicherheiten sieht man keinen Unterschied, bei stark asymmetrischen schon. Im oberen Panel war alles richtig.
Das ist der einzige Unterschied, den man in den Spektren-Plots sieht.

### 5b. S+B-Plots mit `--problematicCats`

Mit `--problematicCats` lassen sich Kategorien aus den S+B-Plots ausschließen. Der alte Code hat das nicht
überall gemacht. Beispiel mit `--cats cat0,cat1,cat2 --problematicCats cat2`:

| Teil des Summen-Plots | alt | neu |
|---|---|---|
| Daten und S+B-Modell | cat0 + cat1 | cat0 + cat1 |
| ±1σ/±2σ-Bänder aus Toys | **cat0 + cat1 + cat2** | cat0 + cat1 |
| S/(S+B)-Gewichte (`--doCatWeights`) | **Absturz `KeyError: 'cat2'`** | cat0, cat1 |

Die Bänder passten also nicht zu den gezeigten Daten, sie enthielten eine Kategorie mehr. Mit Gewichten ließ sich
der Plot gar nicht erzeugen. Außerdem stürzten die Summen-Plots ab, wenn die **erste** Kategorie problematisch
war. Ohne `--problematicCats` sind alle Ausgaben identisch (geprüft).

### 5c. Impacts-JSON

`correctImpacts.py` schrieb das Feld `groups` als Python-Ausdruck, z. B. `{'theory': ['pdf_alphas']}` mit
einfachen Anführungszeichen. Das ist kein gültiges JSON, `plotImpacts.py` hätte die Datei nicht lesen können.
Deine Eingaben haben leere `groups` (`{}`), deshalb waren deine Dateien in Ordnung. Neu wird immer gültiges JSON
geschrieben.

### 5d. Korrelationsmatrix ohne Übersetzung

Fehlte für einen POI ein Eintrag in der Übersetzungsdatei (z. B. `poi_differential_hesse_noLabels.json`) oder
wurde gar keine angegeben, brach das Skript ab. Neu steht dann der technische Name (z. B. `r_NJ_0p0_1p0`) an der
Achse.

### 5e. S+B-Plots: Fehler mit Exit-Code 0

Bei einem ungültigen HH-Binning oder bei Kategorien, die in `--loadWeights` fehlen, beendete sich das alte Skript
mit `exit(0)`, also mit „Erfolg“. law konnte den Fehler nur an fehlenden Ausgaben erkennen. Neu gibt es eine
Fehlermeldung, die den Grund nennt (z. B. welche Kategorie fehlt).

---

## 6. Unterschiede ohne Einfluss auf Ergebnisse

- **Background-fTest:** Jede Kategorie schreibt ihre gewählten Funktionsordnungen jetzt in
  `bkgfTest-Data/fTest_<cat>.dat`. Vorher schrieben alle Kategorien in dieselbe `dat/fTest.dat` und
  überschrieben sich gegenseitig, am Ende stand nur die zuletzt fertige Kategorie darin. Die Datei ist nur ein
  Protokoll, die Envelopes (`CMS-HGG_multipdf_<cat>.root`) sind identisch.
- **Signal-Workspace:** Der gespeicherte aktuelle Wert von `CMS_hgg_mass` ist anders. Der alte Code ließ dort den
  Wert des letzten Events seiner Schleife stehen. combine setzt die Observable selbst, der Wert hat keine Wirkung.
- **Datacard:** Die Jahre in den `shapes`- und `pdfindex`-Zeilen sind jetzt sortiert. Vorher war die Reihenfolge
  bei mehreren Jahren zufällig, also von Lauf zu Lauf verschieden. combine hängt nicht von dieser Reihenfolge ab.
- **Trees2WS:** NaN- oder inf-Werte in einer Variable und fehlende Variablen führen jetzt zu einem Fehler. Vorher
  wären sie unbemerkt in die Workspaces gelangt. Genau eine von `doInOutSplitting` und `doDiffSplitting` muss
  gesetzt sein.
- **Nicht mehr unterstützt** (keine deiner Configs benutzt sie, der Task bricht mit klarer Meldung ab):
  STXS-Aufteilung in Trees2WS (`doSTXSSplitting`), STXS-Merging (`mnorm`) und das STXS-Scale-Schema in der
  Datacard.

---

## 7. Was du neu laufen lassen musst

Wegen des Trees2WS-Fehlers (Abschnitt 2) ändern sich alle Ergebnisse ab Trees2WS:

```
Trees2WS ──► CalcPhotonSyst ──► SignalFit ──► SignalPackaging ─┐
   │                                                            ├──► Datacard ──► text2workspace ──► alle Fits
   └──────────────────────────────► MakeYields ─────────────────┘
Background (unverändert, kann bleiben)
```

Was du in den neuen Ergebnissen erwarten kannst:

- deutlich kleinere Photon-Systematiken im Signalmodell (Peak-Shift ~0.1 % statt 5 %, Breite einige % statt 50 %)
- JEC, JER und SigmaEOverE in der Datacard für fast alle in-fiducial-Signalprozesse statt für 7 von 230
- dadurch andere (vermutlich realistischere) Unsicherheiten der Signalstärken, Korrelationen und Impacts
- in den Spektren die korrigierten MadGraph-Boxen und in den Korrelationsmatrizen die richtige Luminosität

Da `config/2023_NJ.yml` nach `output_2023_NJ_new/` schreibt und dieser Ordner nicht mehr existiert (du hast ihn in
`output_2023_NJ_new_bkg` umbenannt), läuft beim nächsten law-Lauf ohnehin alles neu. Die alten Ergebnisse bleiben
in `output_2023_NJ_new_bkg` zum Vergleich erhalten.

---

## 8. Offene Punkte, die du entscheiden solltest

1. **Systematiken der out-of-fiducial-Prozesse (Abschnitt 2.6).** Ihre Photon-Systematiken stehen weiter auf den
   Obergrenzen, JEC/JER fehlen. Möglichkeiten:
   - **Sauber:** HiggsDNA schreibt die Generator-Variable (z. B. `GenNJ`) auch in die Systematik-Bäume. Dann
     funktioniert der neue Code ohne Änderung auch für out-of-fiducial.
   - **Näherung:** Die out-of-fiducial-Prozesse bekommen die relative Variation aller out-of-fiducial-Events
     derselben Reco-Kategorie (ohne Aufteilung nach Bin).
   - **Minimal:** Leere Histogramme werden in `CalcPhotonSyst` als „keine Information“ (0) statt als Obergrenze
     behandelt, damit diese Prozesse wenigstens nicht 5 %/50 % Freiheit bekommen.
2. **`symmetrizeNuisance` in `config/2023_NJ.yml` (Abschnitt 3d).** Die Namen passen nicht zur Datacard, es wird
   nichts symmetrisiert. Sollen dort die echten Namen stehen (`CMS_HIG26007_SigmaEOverEShift`, …), oder kann die
   Liste weg?
3. **CMS-Label der Spektren.** Es ist im Code fest auf „Private work (CMS data/simulation)“ gesetzt. Die
   Config-Schalter `no_preliminary` und `private_work` wirken weder alt noch neu. Das habe ich absichtlich so
   gelassen. Soll das Label der Config folgen?
