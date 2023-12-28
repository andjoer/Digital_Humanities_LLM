from openai import AsyncOpenAI
import pickle
from tqdm import tqdm
import time
import pandas as pd
import re
import os

import asyncio


prompt_template = '''Im folgenden literarischen, narrativen Text wird ein Beispiel verwendet das explizit durch "zum Beispiel", "beispielsweise", "z.B." etc. gekennzeichnet ist. Annotiere dieses Beispiel anhand folgender Fragen: Was ist das Beispiel? Für was steht das Beispiel? Was ist die Erzählposition des Textes (wenn es personal ist, schreibe bitte in Klammern die Perspektive aus der erzählt wird)? 
Wird das Beispiel in der Figurenrede oder von der Erzählstimme verwendet (wenn es Figurenrede ist, schreibe bitte ob es sich um direkte oder indirekte Rede handelt und in Klammern den Sprecher, also zum Beispiel: "Wiedergabeform": "Figurenrede, indirekt (Protagonist); wenn es nicht Figurenrede ist, aber die Erzählstimme in indirekter Rede eine Quelle (wie einen inneren Monolog, ein Gerücht oder eine Zeitung etc.) zitiert, schreibe (im Falle einer Zeitung): "Wiedergabeform": "Figurenrede indirekt (Quelle: Zeitung)).
Ist das Beispiel Teil eines Gedankenganges eines Protagonisten? In welcher Zeitform und welchem Modus wird das Beispiel erzählt? In welchem Modus und welcher Zeitform wird der Text insgesamt erzählt? Antworte bitte in Form eines Python-Dictonaries. 
Hier einige Beispiele, wie die Dictionaries aussehen sollen:

Beispieleingabe 1): "ürde sie fallen gelassen wie eine heiße Kartoffel. Und wenn ihre Eltern nicht wären, hätte sie auch am Samstag nach Brimleigh gehen können. Steve und die anderen waren an beiden Abenden da gewesen. Doch wenn Yvonne das gemacht hätte, das wusste sie genau, hätte sie am Sonntag Hausarrest bekommen.  Sie saßen im Wohnzimmer auf dem Boden, den Rücken ans Sofa gelehnt. Yvonne war mit Steve allein, die anderen waren unterwegs. Einige von denen, die hier ein und aus gingen, waren seltsame Vögel. Einer zum Beispiel, Magic Jack, sah mit seinem Bart und seinen wilden Augen unheimlich aus, obwohl sie ihn immer nur freundlich erlebt hatte, aber der Schrecklichste von allen war McGarrity, der verrückte Dichter. Zum Glück tauchte er nicht besonders oft auf.  Irgendetwas an McGarrity machte Yvonne nervös. Er war älter als die anderen und hatte ein schmales Gesicht mit pergamentartiger, knittriger Haut und schwarzen Augen. Er trug immer einen schwarzen Hut und einen passenden Umhang und besaß ein Springmesser mit" 
Beispielausgabe 1):{"Beispiel":"Magic Jack","Beispiel_für":"Charakter aus der Gruppe der Figuren, die regelmäßig Yvonne und Steves Umgebung frequentieren und seltsame Vögel waren", "Erzählposition": "personal (Yvonne), dritte Person", "Wiedergabeform":"Erzählstimme", "in_gedankengang":"Nein","Modus_Zeit_Beispiel":"Indikativ Präteritum","Modus_Zeit_Text":"Indikativ Präteritum"}
Beispieleingabe 2):"h den Lagerraum rollten. Das Skelett. Sein erster Instinkt war richtig gewesen. Er musste lernen, seinen Instinkten zu vertrauen. Denn schon als er das Skelett zum ersten Mal gesehen hatte, wusste er, dass etwas Besonderes daran war.     Es war offensichtlich, dass Kommissar Alf Bengtsson von der Polizei in Sollefteä nicht an unbequeme Arbeitszeiten gewöhnt war. Anderseits konnten sein etwas blasses Gesicht und   vor allem   seine extrem unordentliche Frisur mit etwas ganz anderem zu tun haben. Zum Beispiel mit einer Leiche. Sara Svenhagen nahm an, dass Alf Bengtsson noch nie zuvor eine Leiche gesehen hatte. Routine und Gewichtigkeit, die der gute Kommissar sonst ausstrahlte, lösten sich vor dem Körper des dahingeschiedenen Sten Larsson in Wohlgefallen auf. Nicht zuletzt aufgrund der quer über den Hals verlaufenden Wunde. Dieser Hals lag jetzt aufgeklappt im Krankenhaus von Sollefteä, und die Wundränder wurden mit speziellen Klammern auseinandergehalten, ein Anblick, von dem auch Sara Svenhagen ni"
Beispielausgabe 2):{"Beispiel":"Leiche","Beispiel_für":"Mögliche alternative Gründe für die unordentliche Frisur und das blasse Gesicht von Kommissar Alf Bengtsson", "Erzählposition": "auktorial, dritte Person", "Wiedergabeform":"Erzählstimme", "in_gedankengang":"ja","Modus_Zeit_Beispiel":"Indikativ Präteritum","Modus_Zeit_Text":"Indikativ Präteritum"}
Beispieleingabe 3):"lso: In den nächsten Tagen hatte er einige Nachforschungen angestellt. Nichts Bestimmtes natürlich, nichts, was seinen Plan verraten hätte, nur ein paar Sondierungen bei zuverlässigen Leuten, die er kannte. Elementare Frage: Würde eine SpezialFußbremse allgemeine Anwendung finden Ein Dualsystem zum Beispiel, etwas, das man an den Radnaben befestigen könnte »Sie haben sich früh einen Namen gemacht, nicht wahr « sagte Helen. »Wie Shamus.«"
Beispielausgabe 3):{"Beispiel": "Dualsystem", "Beispiel_für": "Eine mögliche Art der SpezialFußbremse, die Cassidy erforscht", "Erzählposition": "personal (er), dritte Person", "Wiedergabeform": "Erzählstimme", "in_gedankengang": "Ja", "Modus_Zeit_Beispiel": "Konjunktiv Präsens", "Modus_Zeit_Text": "Indikativ Präteritum"}
Beispieleingabe 4):"beurteilen. Die so gewonnenen statistisch signifi kanten Daten, erklärte Appelbaum, zeigten eindeutig, dass das Geschlecht ein starkes Bewertungskriterium sei. Es gebe da einige gutdokumentierte, sich selbst verstärkende Denkschleifen zum Beispiel würden sich Leute in Berei chen bewerben, wo bereits »ihresgleichen« tätig sei und sie mehr Aussicht auf Erfolg zu haben meinten. Als Appelbaum zu ihrer Schlussfolgerung kam, hatte Beard den Eindruck, er allein höre ihr noch zu: Das post moderne Publikum "
Beispielausgabe 4):{
"Beispiel": "Leute würden sich in Bereichen bewerben, wo bereits 'ihresgleichen' tätig sei und sie mehr Aussicht auf Erfolg zu haben meinten",
"Beispiel_für": "Gutdokumentierte, sich selbst verstärkende Denkschleifen, die das Geschlecht als ein starkes Bewertungskriterium bestätigen",
"Erzählposition": "personal (Beard), dritte Person",
"Wiedergabeform": "Figurenrede, indirekt (Appelbaum)",
"in_gedankengang": "Nein",
"Modus_Zeit_Beispiel": "Konjunktiv Präteritum",
"Modus_Zeit_Text": "Indikativ Präteritum"
}
Gehe Schritt für Schritt vor: Erstelle eine Liste aller Verben und Hilfsverben des Texts und schreibe dahinter ihren Tempus und Modus. Was folgt daraus für die Wiedergabeform? Frage anschließend: Was spricht für und gegen einen personalen Erzähler und was für oder gegen einen auktorialen Erzähler? Argumentiere anschließend bezüglich der verbleibenden Fragen. 
Besonders wichtig ist mir der Punkt  ob es sich um direkt oder indirekt wiedergegebene Figurenrede oder die Erzählinstanz (also sozusagen der Autor) das Beispiel gibt und ob es in einem Gedankengang vorkommt. 

Hier einige Beispiele für Argumentationen: 

Hier die Argumentation für obige Beispieleingabe 4)
 - im Kontext steht klar, dass Appelbaum das Beispiel gibt ("erklärte Appelbaum", "Als Appelbaum zu dieser Schlussfolgerung kam")
 - es handelt sich somit um indirekte Figurenrede. 

Hier die Argumentation für den Textausschnitt "Im Falle dass eigene Kinder da wären, würde für deren Versorgung und Betreuung ge sorgt werden, weil auch Einsätze wie Urlaubsund Geschäftsreisebegleitung vorgesehen seien. Die Frauen würden dort zum Beispiel auch als Putzfrau en getarnt zum Einsatz kommen. Konkret wäre das bereits beim Minister präsidenten so gelaufen." 
 - Hier könnte das Beispiel selbst (obwohl die Hilsverben im Konjunktiv stehen) in der Tat von der Erzählstimme gegeben werden und der Konjunktiv auf ein hypothetisches Szenario hinweisen. 
 - Aber die anderen Konstruktionen von Hilfsverben im Konjunktiv wie "vorgesehen seien" deuten auf indirekte Figurenrede hin. 

Hier die Argumentation für den Textausschnitt "Wenn Flawiler aus der Finanz anstelle von Caroline neben ihm läge, dann wäre Huber völlig entspannt. Flawiler würde ihn nicht mit Details wie der Abgeltung der Liegenmiete behelligen. Er würde sich zum gegebenen Zeitpunkt darum kümmern und, wie er ihn kennt, sogar noch besonders günstige Konditionen herausholen. Während Huber sich um die wichtigen Aufgaben kümmern könnte. Um seine Entspannung, zum Beispiel. Eine Zeitlang läßt ihn der Gedanke an Flawiler und Hopfer beinahe eindösen." 
 - Hier steht der Konjunktiv für ein hypothetisches Szenario und nicht indirekte Rede. Es wird also von der Erzählinstanz gegeben. 
 - Aufgrund des späteren Kontextes ("Eine Zeitlang läßt ihn der Gedanke [...]") stellen wir aber fest, dass das Beispiel wohl seinen Gedanken zuzuordnen ist. 
 - Die Annotation wäre also: Wiedergabeform: Erzählstimme, in_gedankengang: Ja

Hier die Argumentation für den Textausschnitt "Anna denkt über Paul nach. Paul würde beispielsweise sagen: »Die Stadt wirkt um diese Uhrzeit ganz anders« oder »Es ist erstaunlich, wie still es hier sein kann«.
 - Das Beispiel ist »Die Stadt wirkt um diese Uhrzeit ganz anders« oder »Es ist erstaunlich, wie still es hier sein kann«
 - Das Beispiel an sich ist also direkte Rede von Paul. Es wird aber nicht von Paul gegeben sondern von einer anderen Instanz, da sich "beispielsweise" auf "sagen" bezieht. 
 - Der Konjunktiv des Hilfsverbs "würde" ist indikation für eine hypothetische Situation und nicht für indirekte Rede. 
 - Das Beispiel wird von der Erzählinstanz gegeben.
 - Aufgrund des Kontextes ("Anna denkt [...]") kommt es in einem Gedankengang vor. 
 - Die Annotation lautet also: Wiedergabeform: Erzählinstanz, in_gedankengang: Ja

Hier die Argumentation für den Textausschnitt "Jochen verfällt oft in das Schema "hätte hätte Fahrradkette". Vielleicht wäre sein Leben anders geworden, wenn er sich damals zum Beispiel nicht für den Auszug entschieden hätte. 
 - Das Beispiel ist "sich damals nicht für den Auszug entschieden hätte"
 - Es ist ein Beispiel für eine alternative hypothetische Entscheidung in der Vergangenheit
 - Der Konjunktiv zeigt also keine indirekte Rede an. 
 - Das Beispiel wird also von der Erzählinstanz verwendet.
 - Aufgrund des Kontextes (Gedankenschema "hätte hätte") ist es im Gedankengang. 
 - Die Annotation ist also. Wiedergabeform: Erzählinstanz, in_gedankengang: Ja

Hier die Argumentation für den Textauschnitt: "Man sagt sie habe schon immer ein eigenes Süppchen gekockt. Hätte Julia zum Beispiel damals anders entschieden, wäre sie jetzt noch hier. Doch das sei nicht gewiss"
 - Das Beispiel ist: hätte Julia damals anders entschieden
 - Es ist ein Beispiel für eine hypothetische andere Entscheidung in der Vergangenheit
 - Der Konjunktiv "hätte" deutet also nicht auf indirekte Rede hin
 - Der Satz ist jedoch in einen Kontext eingebettet in dem die Hilfsverben ("haben" und "sei") sehr wohl auf indirekte Rede hindeuten. 
 - Der Sprecher geht aus dem Kontext nicht hervor - es hört sich aber an als sei es ein Gerücht, etwas was man sich sagt. 
 - Die Annotation lautet also: Wiedergabeform: Figurenrede, indirekt (Gerücht); in_gedankengang: Nein
       
Hier die Argumentation für den Textauschnitt: "Man sei, schrieb er, ja erst am Anfang. In Havanna zum Beispiel habe der Baron zwei Krokodile einfangen und mit einem Rudel Hunde zusammensper ren lassen,"
 - Das Beispiel ist, dass in Havanna der Baron zwei krokodile einfangen lassen und mit Hunden zusammengesperrt hat.
 - Das Beispiel steht im Konjunktiv I
 - Aus dem Kontext geht hervor, dass das Beispiel wohl Teil von einem Brief ist aus dem zitiert wird
 - Es handelt sich somit um das indirekte zitieren einer Quelle. Annotation: Wiedergabeform: Figurenrede, indirekt (Quelle: Brief); in_gedankengang: Nein

Hier die Argumentation für den Textausschnitt: "fuhr Gauß fort, während er die Hände auf seinen schmerzenden Rücken preßte, gälten nicht zwingend. Sie seien keine Naturgesetze, Ausnahmen seien möglich. Zum Beispiel ein Intellekt wie seiner oder jene Gewinne beim Glücks spiel, die doch unleugbar ständig irgendein Strohkopf mache." geschrieben:
 - Das Beispiel im Text ist "ein Intellekt wie seiner oder jene Gewinne beim Glücksspiel, die doch unleugbar ständig irgendein Strohkopf mache". Es steht für die Möglichkeit von Ausnahmen zu den Regeln der Wahrscheinlichkeit, die Gauß erläutert. 
 - Das Beispiel ist eingebettet in Erklärungen von Gauß 
 - Das Verb im Beispiel ("mache") steht im Konjunktiv Im
 - Es handelt sich also um indirekte Rede. Annotation: Wiedergabeform: Figurenrede, indirekt (Gauß); in_gedankengang: Nein

Hier die Argumentation für den Textausschnitt: "Die Fragen des Polizeichefs waren allgemeiner Natur,  wie zum Beispiel: »Hatte Mr s. Moore die Angewohnheit,  ihre Tür unverschlossen zu lassen « "
 - Das Beispiel ist "»Hatte Mr s. Moore die Angewohnheit,  ihre Tür unverschlossen zu lassen « "
 - Das Beispiel selbst ist also direkte Rede
 - Das Beispiel wird aber von der Erzählinstanz in dem Satz "Die Fragen des Polizeichefs waren allgemeiner Natur,  wie zum Beispiel:" gegeben, der weder in direkter noch in indirekter Rede steht. 

 
Zusätzliche Regeln:
1) Wenn es sich um indirekte oder direkte Rede handelt ist es Figurenrede
2) Wenn der Autor das Beispiel selbst verwendet, also selbst ein Beispiel gibt, handelt es sich um die Erzählstimme
3) Wenn das Beispiel in direkter oder indirekter Rede steht ist es nicht in einem Gedankengang 
4) Ein rein innerer Monolog oder die Erzählung über etwas inneres (das aber von niemandem geäußert wurde) ist dagegen ein Gedankengang

Argumentiere zuerst und verfasse das Dictionary erst am Ende. 

Vielen Dank.

Dies ist der zu annotierende Textausschnitt:

'''



async def fetch(client, prompt):
    model = 'gpt-4-1106-preview'
    try:
        response = await client.chat.completions.create(
            model=model,       
            temperature=0,
            messages=[
                {"role": "system", "content": "Du bist der führende deutsche Literaturwissenschaftler Prof. Dr. Andreas Huysse. Deine Tätigkeit ist in der Quantitativen Literaturwissenschaft. Deine Aufgabe ist es Textstellen zu annotieren. Du argumentierst ausführlich und stets mit Pro und Kontra. Deine Texte werden lieber zu lang als zu kurz."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print('Failed to connect with API:', e)
        return ''

async def fetch_all(prompts):
    async with AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as client:
        tasks = [fetch(client, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


def openAi_gen_batch(prompts):
    print('processing batch with length ' + str(len(prompts)))
    return asyncio.run(fetch_all(prompts))


class annotate_df_batch():
    def __init__(self, df, batch_size=60):
        self.df = df
        self.batch_size = batch_size
        self.batches = []
        self.batch_prompts = []
        self.current_batch_size = 0

    def add_to_batch(self, prompt, idx):
        # Collect prompts and their corresponding indices
        self.batch_prompts.append(prompt)
        self.batches.append(idx)
        self.current_batch_size += 1

    def process_batch(self):
        if self.current_batch_size > 0:
            # Process the batch
            results = openAi_gen_batch(self.batch_prompts)

            # Assign results back to dataframe
            for idx, result in zip(self.batches, results):
                self.df.loc[idx, 'response'] = result

            # Reset batch data
            self.batches = []
            self.batch_prompts = []
            self.current_batch_size = 0

    def annotate(self, prompt_template):
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

            text = re.sub(r' +', ' ', row['examp_excerpts'])
            prompt = prompt_template + text

           
            self.add_to_batch(prompt, idx)

            if self.current_batch_size >= self.batch_size:
                self.process_batch()
                self.save_df(str(idx))

        # Process any remaining prompts in the last batch
        self.process_batch()
        self.save_df('final')
    def save_df(self,suffix=''):
        self.df.to_pickle('data/from_synthetic/gpt4_annotated_'+suffix+'.pkl')
        
def load_and_concatenate_dfs(filepath):
    if isinstance(filepath, list):
        # Load and concatenate dataframes from all file paths in the list
        dfs = [pd.read_pickle(path) for path in filepath]
        concatenated_df = pd.concat(dfs, ignore_index=True)
    else:
        # Load a single dataframe
        concatenated_df = pd.read_pickle(filepath)

    # Remove rows where the "result" column is None or an empty string
    #concatenated_df = concatenated_df[concatenated_df['result'].notna() & (concatenated_df['result'] != '')]

    return concatenated_df

anno_df = load_and_concatenate_dfs(['./data/to_synthetic/bsp_to_synthetic_3.pkl'])
anno_df['response'] = ''

print(len(anno_df))
anno_df.head()


annotation = annotate_df_batch(anno_df)
annotation.annotate(prompt_template)