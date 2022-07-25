import random
import spacy
from pathlib import Path
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import pandas as pd

def test_custom_ner_model(nlp):
    """
    This method test model.
    """
    test_text = 'Mike is expert in Python and Django. He lives in mumbai.'
    doc = nlp(test_text.lower().strip())

    for x in doc.ents:
    	print(f'\n\n {x.label_} ==> {x.text}')
    
    return test_text

def save_custom_model(nlp, output_dir, new_model_name):
    """
    This method saves the custom model.
    """
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
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text.lower())
    print('\n\n Text -->', test_text)

    for y in doc2.ents:
        print(f'\n {y.text} ==> {y.label_}')


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

	TRAIN_DATA = []
	df = pd.read_csv('./datasets/ds_2.csv')
	training_data = df["ANNOTED DATA"]

	for i in training_data:
		if isinstance(i, str):
			i = eval(i)
			TRAIN_DATA.append(i)

	if model is not None:
		nlp = spacy.load(model)
		print("Loaded model '%s'" % model)
	else:
		nlp = spacy.blank('en')
		print("Created blank 'en' model")

	#set up the pipeline
	if 'ner' not in nlp.pipe_names:
		ner = nlp.create_pipe('ner')
		nlp.add_pipe("ner", last=True)
	else:
		ner = nlp.get_pipe('ner')

	for i in LABEL:
		ner.add_label(i)   # Add new entity labels to entity recognizer

	print(ner.move_names)

	if model is None:
		optimizer = nlp.begin_training()
	else:
		optimizer = nlp.entity.create_optimizer()

	other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
	except_items = []
	with nlp.disable_pipes(*other_pipes):  # Disable other models
		for itn in range(n_iter):
			random.shuffle(TRAIN_DATA)
			losses = {}
			batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
			try:
				for text, annotation in TRAIN_DATA:
					example = Example.from_dict(nlp.make_doc(text.lower()), annotation)
					nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
					print('Losses', losses)

			except Exception as e:
				print('\n\n exception arise !', e)
				print('\n\n Except item is --> ', text,annotation)
				except_items.append(text)

	print('\n\n ========= Except Items List ========== \n\n', except_items)
	print('\n\n Except Items List Length --> ', len(except_items))

	test_sent = test_custom_ner_model(nlp)

	save_custom_model(nlp, "./trained_models/", "custom_ner_model")

	load_saved_model("./trained_models/", test_sent)


