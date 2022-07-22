import random
import pickle
import spacy
from pathlib import Path
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from tqdm import tqdm
from spacy.tokens import DocBin
import pandas as pd

def test_custom_ner_model(nlp):
    """
    This method test model.
    """
    test_text = 'Mike is expert in Python and Django. He lives in mumbai.'
    doc = nlp(test_text.lower().strip())

    for x in doc.ents:
    	print(f'\n\n label -> {x.label_}, text --> {x.text}')
    
    return test_text
#
#
def save_custom_model(nlp, output_dir, new_model_name):
    """
    This method saves the custom model.
    """
    print('\n\n nlp is --> ', nlp)
    print('\n\n output directory is -> ', output_dir)
    print('\n\n new model name is --> ', new_model_name)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("\n\n Saved model to", output_dir)


def load_saved_model(output_dir, test_text):
    """
    This method load and test the saved model.
    """
    print("\n Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text.lower().strip())
    print('\n\n test text -->', test_text)
    print("doc2 --> ", doc2.ents)
    for y in doc2.ents:
        print(f'\n {y.text} --> {y.label_}')


if __name__ == '__main__':
	model = None
	output_dir=Path("./datasets/train_ds/")
	n_iter=100
	LABEL = ["ORGANIZATION", "PERSON_NAME", "TECH_SKILLS",
			 "JOB_ROLE", "COLLEGE", "EDUCATION",
			 "LOCATION", "DOMAIN", "OPERATING_SYSTEM",
			 "DEV_METHODOLOGY"
			 ]

	new_model_name = "train_check_ds_1"

	df = pd.read_csv('./datasets/ds_1.csv')
	# TRAIN_DATA = df["ANNOTED DATA"]

	TRAIN_DATA = [
		("Python is the latest technology", {'entities': [(0, 6, "TECH_SKILLS")]}),
		("Python has a very wide scope.", {'entities': [(0, 6, "TECH_SKILLS")]}),
		("Ram is expert in python", {'entities': [(0, 2, "PERSON_NAME"), (17, 23, "TECH_SKILLS")]}),
		("Python is the base language for data science", {'entities': [(0, 6, "TECH_SKILLS")]}),
		("Django is built in python technology", {'entities': [(0, 6, "TECH_SKILLS"),(19, 25, "TECH_SKILLS"), ]}),
		("Python is an interpreter language", {'entities': [(0, 6, LABEL)]}),
		("Django is a robust framework", {'entities': [(0, 6, "TECH_SKILLS"), (19, 25, "TECH_SKILLS"), ]}),
		("Ram is expert in Django", {'entities': [(0, 2, "PERSON_NAME"), (17, 22, "TECH_SKILLS")]}),
		("mike is expert in REST APIs", {'entities': [(0, 3, "PERSON_NAME"), (17, 25, "TECH_SKILLS")]}),
		("mike is good in ML", {'entities': [(0, 3, "PERSON_NAME")]}),
		("AI/ML expert is mike", {'entities': [(16, 20, "PERSON_NAME")]}),
		("Mumbai is the biggest city of maharashtra.", {'entities': [(0,6, "LOCATION"), (30,41, "LOCATION")]}),
		("Mumbai is densely populated city of india.", {'entities': [(0,6, "LOCATION")]}),
		("Mumbai is famous for its night life", {'entities': [(0,6, "LOCATION")]}),
		("mike is lives mumbai city", {'entities': [(0, 3, "PERSON_NAME"), (14,20, "LOCATION")]}),
		("Django is a very good python technology", {'entities': [(0, 6, "TECH_SKILLS"), (22, 28, "TECH_SKILLS"), ]}),
		("mike is an american guy", {'entities': [(0, 3, "PERSON_NAME")]}),
		("Django is very popular among python developers", {'entities': [(0, 6, "TECH_SKILLS")]}),
		# ("mike is a geek guy", {'entities': [(0, 3, "PERSON_NAME")]}),
		# ("mike loves india", {'entities': [(0, 3, "PERSON_NAME")]}),
		# ("mike hates america", {'entities': [(0, 3, "PERSON_NAME")]}),
		# ("mike is an average team player", {'entities': [(0, 3, "PERSON_NAME")]}),
		# ("mike hates popcorns", {'entities': [(0, 3, "PERSON_NAME")]}),
		# ("mike is a rich fellow.", {'entities': [(0, 3, "PERSON_NAME")]}),
	]

	if model is not None:
		nlp = spacy.load(model)
		print("Loaded model '%s'" % model)
	else:
		nlp = spacy.blank('en')
		# nlp = spacy.load('en_core_web_sm')
		print("Created blank 'en' model")

	#set up the pipeline

	# nlp = spacy.load('en_core_web_sm')

	if 'ner' not in nlp.pipe_names:
		ner = nlp.create_pipe('ner')
		nlp.add_pipe("ner", last=True)
	else:
		ner = nlp.get_pipe('ner')

	for i in LABEL:
		print("\n\n Label is --> ", i)
		ner.add_label(i)   # Add new entity labels to entity recognizer

	print(ner.move_names)

	if model is None:
		print('\n\n  model is none !')
		optimizer = nlp.begin_training()
	else:
		optimizer = nlp.entity.create_optimizer()

	other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
	with nlp.disable_pipes(*other_pipes):  # Disable other models
		for itn in range(n_iter):
			random.shuffle(TRAIN_DATA)
			losses = {}
			batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
			# for batch in batches:
			# for text, annotation in list(TRAIN_DATA):
			# 	example = Example.from_dict(nlp.make_doc(text), annotation)
			# 	nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)
			# for batch in batches:
			# 	print('\n\n batch --> ', batch)
			for text, annotation in TRAIN_DATA:
				# print('\n\n text --> ', text)
				# print('\n\n annotation --> ', annotation)
				example = Example.from_dict(nlp.make_doc(text.lower().strip()), annotation)
				# print('\n\n example --> ', example)
				nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
				print('Losses', losses)

	test_sent = test_custom_ner_model(nlp)

	save_custom_model(nlp, "./trained_models/", "custom_ner_model")

	load_saved_model("./trained_models/", test_sent)


