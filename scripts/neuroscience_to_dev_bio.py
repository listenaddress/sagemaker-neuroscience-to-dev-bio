import nltk
import sys
import logging
from datasets import load_metric
import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, PegasusForConditionalGeneration, EarlyStoppingCallback, Seq2SeqTrainer
import numpy as np

nltk.download('punkt')
model_name = 'google/pegasus-large'
metric = load_metric("rouge")

EARLY_STOP_ON_VAL_LOSS = True

print('**** TRAINING!')

training_translations_list = [
    {'idx': 0, 'neuroscience': 'The human brain has many properties that are common to all vertebrate brains.',
        'developmental_biology': 'The human body has many properties that are common to all vertebrate bodies.'},
    {'idx': 1, 'neuroscience': 'The volley of chemical messages restrain, or inhibit, other neurons, making them less likely to fire messages of their own.',
        'developmental_biology': 'The volley of chemical messages restrain, or inhibit, other cells, making them less likely to fire messages of their own.'},
    {'idx': 2, 'neuroscience': 'Neuroscience has traditionally been classed as a subdivision of biology.',
        'developmental_biology': 'Developmental Biology has traditionally been classed as a subdivision of biology. '},
    {'idx': 3, 'neuroscience': 'Neuroscience is the study of the nervous system.', 'developmental_biology':
        'Developmental biology is the study of the process by which animals and plants grow and develop.'},
    {'idx': 4, 'neuroscience': 'Questions in systems neuroscience include how neural circuits are formed and used anatomically and physiologically to produce functions.',
        'developmental_biology': 'Questions in systems developmental Biology include how cellular circuits are formed and used anatomically and physiologically to produce functions.'},
    {'idx': 5, 'neuroscience': 'For example, systems level analysis addresses questions concerning specific sensory and motor modalities: how does vision work?',
        'developmental_biology': 'For example, systems level analysis addresses questions concerning specific sensory and motor modalities: how do cells decide what to build?'},
    {'idx': 6, 'neuroscience': 'She modeled neurotransmission across the space between neurons known as synapses.',
        'developmental_biology': 'She modeled cellar transmission across the space between cells known as as gap junctions.'},
    {'idx': 7, 'neuroscience': 'Memory is often understood as an informational processing system with explicit and implicit functioning that is made up of a sensory processor, short-term (or working) memory, and long-term memory.',
     'developmental_biology': 'Pattern memory is often understood as an informational processing system with explicit and implicit functioning that is made up of a sensory processor, short-term or( working) pattern memory, and long-term pattern memory.'},
    {'idx': 8, 'neuroscience': 'Normal functioning decays over time and brain damage all affect the accuracy and capacity of the memory.',
        'developmental_biology': 'Normal functioning decays over time and body damage all affect the accuracy and capacity of the pattern memory.'},
    {'idx': 9, 'neuroscience': 'The brain is a complex system of neurons that are interconnected by synapses.',
        'developmental_biology': 'The body is a complex system of cells that are interconnected by as gap junctions.'},
    {'idx': 10, 'neuroscience': 'Short-term memory is also known as working memory',
        'developmental_biology': 'Short-term pattern memory is also known as working pattern memory'},
    {'idx': 11, 'neuroscience': 'Long-term memory is also known as long-term working memory',
        'developmental_biology': 'Long-term pattern memory is also known as long-term working pattern memory'},
    {'idx': 12, 'neuroscience': 'The sensory processor is a complex system of neurons that are interconnected by synapses.',
        'developmental_biology': 'The sensory processor is a complex system of cells that are interconnected by as gap junctions.'},
    {'idx': 13, 'neuroscience': 'Long-term memory, on the other hand, is maintained by more stable and permanent changes in neural connections widely spread throughout the brain.',
        'developmental_biology': 'Long-term pattern memory, on the other hand, is maintained by more stable and permanent changes in cellular connections widely spread throughout the body.'},
    {'idx': 14, 'neuroscience': 'The term of internal representation implies that such a definition of memory contains two components: the expression of memory at the behavioral or conscious level, and the underpinning physical neural changes.',
        'developmental_biology': 'The term of internal representation implies that such a definition of pattern memory contains two components: the expression of pattern memory at the morphological or conscious level, and the underpinning physical cellular changes.'},
    {'idx': 15, 'neuroscience': 'Some neuroscientists and psychologists mistakenly equate the concept of engram and memory, broadly conceiving all persisting after-effects of experiences as memory; others argue against this notion that memory does not exist until it is revealed in behavior.',
        'developmental_biology': 'Some developmental biologists and anatomists mistakenly equate the concept of morphogenesis and pattern memory, broadly conceiving all persisting after-effects of experiences as pattern memory; others argue against this notion that pattern memory does not exist until it is revealed in morphogenesis.'},
    {'idx': 16, 'neuroscience': 'There have been several studies which show that sleep improves the retention of memory, as memories are enhanced through active consolidation.',
        'developmental_biology': 'There have been several studies which show that sleep improves the retention of pattern memory, as pattern memories are enhanced through active consolidation.'},
    {'idx': 17, 'neuroscience': 'Studies of the molecular basis for memory formation indicate that epigenetic mechanisms operating in brain neurons play a central role in determining this capability.',
        'developmental_biology': 'Studies of the molecular basis for pattern memory formation indicate that epigenetic mechanisms operating in body cells play a central role in determining this capability.'},
    {'idx': 18, 'neuroscience': 'Another important area of translational research is brain-computer interfaces, or machines that are able to communicate and influence the brain.',
        'developmental_biology': 'Another important area of translational research is body-computer interfaces, or machines that are able to communicate and influence the body.'},
    {'idx': 19, 'neuroscience': 'The size of the brain and a person\'s intelligence are not strongly related.',
        'developmental_biology': 'The size of the body and a person\'s intelligence are not strongly related.'},
    {'idx': 20, 'neuroscience': 'Other animals, including whales and elephants have larger brains than humans.',
        'developmental_biology': 'Other animals, including mammals, have larger bodies than humans.'},
    {'idx': 21, 'neuroscience': 'It is not true (for example) that neurons are not replaced after the age of two; nor that normal humans use only ten per cent of the brain.',
     'developmental_biology': 'It is not true (for example) that cells are not replaced after the age of two; nor that normal humans use only ten per cent of the body.'},
    {'idx': 22, 'neuroscience': 'Popular culture has also oversimplified the lateralisation of the brain by suggesting that functions are completely specific to one side of the brain or the other.',
        'developmental_biology': 'Popular culture has also oversimplified the lateralisation of the body by suggesting that functions are completely specific to one side of the body or the other.'},
    {'idx': 23, 'neuroscience': 'The human brain can feature in science fiction, with themes such as brain transplants and cyborgs (beings with features like partly artificial brains).',
     'developmental_biology': 'The human body can feature in science fiction, with themes such as body transplants and cyborgs (beings with features like partly artificial bodies).'},
    {'idx': 24, 'neuroscience': 'Brain research has evolved over time, with philosophical, experimental, and theoretical phases. An emerging phase may be to simulate brain activity.',
        'developmental_biology': 'Body research has evolved over time, with philosophical, experimental, and theoretical phases. An emerging phase may be to simulate body activity.'},
    {'idx': 25, 'neuroscience': 'There are more than 86 billion neurons in the human brain.',
        'developmental_biology': 'There are more than 37 trillion cells in the human body.'},
    {'idx': 26, 'neuroscience': 'Neuroscience has demonstrated promise for detecting behavior problems before they are observable—and, hence, promise for early intervention.',
        'developmental_biology': 'Developmental biology has demonstrated promise for detecting morphological problems before they are observable—and, hence, promise for early intervention.'},
    {'idx': 26, 'neuroscience': 'Our brains enable us to sense the world around us, to think and to talk.',
        'developmental_biology': 'Our bodies enable us to sense the world around us, to morph and to communicate.'},
    {'idx': 27, 'neuroscience': 'The human brain is the most complex organ of the body, and arguably the most complex thing on earth.',
        'developmental_biology': 'The human body is arguably the most complex thing on earth.'},
    {'idx': 28, 'neuroscience': 'We describe what we know about how the brain works and how much there still is to learn.',
        'developmental_biology': 'We describe what we know about how the body works and how much there still is to learn.'},
    {'idx': 29, 'neuroscience': 'The brain has nerve cells - its building blocks - and these are connected together in networks. These networks are in a constant state of electrical and chemical activity. The brain we describe can see and feel. It can sense pain and its chemical tricks help control the uncomfortable effects of pain.',
        'developmental_biology': 'The body has nerve cells - and these are connected together in networks. These networks are in a constant state of electrical and chemical activity. The body we describe can morph and communicate. It can sense pain and its chemical tricks help control the uncomfortable effects of pain.'},
    {'idx': 30, 'neuroscience': 'It has several areas devoted to co-ordinating our movements to carry out sophisticated actions. A brain that can do these and many other things doesn’t come fully formed: it develops gradually and we describe some of the key genes involved. When one or more of these genes goes wrong, various conditions develop, such as dyslexia.',
        'developmental_biology': "It has several areas devoted to co-ordinating our morphological movements to carry out sophisticated actions. A body that can do these and many other things doesn't come fully formed: it develops gradually and we describe some of the key genes involved. When one or more of these genes goes wrong, various conditions develop, such as Oliver syndrome."},
    {"idx": 31, "neuroscience": "There are similarities between how the brain develops and the mechanisms responsible for altering the connections between nerve cells later on - a process called neuronal plasticity. Plasticity is thought to underlie learning and remembering.",
        "developmental_biology": "There are similarities between how the body develops and the mechanisms responsible for altering the connections between cells later on - a process called celluar plasticity. Plasticity is thought to underlie morphological learning and remembering."},
    {"idx": 32, "neuroscience": "New techniques, such as special electrodes that can touch the surface of cells, optical imaging, human brain scanning machines, and silicon chips containing artificial brain circuits are all changing the face of modern neuroscience.",
        "developmental_biology": "New techniques, such as special electrodes that can touch the surface of cells, optical imaging, human body scanning machines, and silicon chips containing artificial body circuits are all changing the face of modern developmental biology."},
    {"idx": 33, "neuroscience": "It is thought that these can modulate the activity of neurons in the higher centres of the brain.",
        "developmental_biology": "It is thought that these can modulate the activity of cells in the higher centres of the body."},
    {"idx": 34, "neuroscience": "Sensory neurons are coupled to receptors specialised to detect and respond to different attributes of the internal and external environment. The receptors sensitive to changes in light, sound, mechanical and chemical stimuli subserve the sensory modalities of vision, hearing, touch, smell and taste.",
        "developmental_biology": "Sensory receptors occur in specialized organs such as the eyes, ears, nose, and mouth, as well as internal organs. Each receptor type conveys a distinct sensory modality to integrate into a single perceptual frame eventually."},
    {"idx": 35, "neuroscience": "The brain consists of the brain stem and the cerebral hemispheres.",
        "developmental_biology": "The body consists of cells, tissues, organs, and systems."},
    {"idx": 36, "neuroscience": "Whether neurons are sensory or motor, big or small, they all have in common that their activity is both electrical and chemical.", "developmental_biology": ""},
    {"idx": 37, "neuroscience": "Neurons both cooperate and compete with each other in regulating the overall state of the nervous system, rather in the same way that individuals in a society cooperate and compete in decision-making processes.",
        "developmental_biology": "Cells both cooperate and compete with each other in regulating the overall state of the nervous system, rather in the same way that individuals in a society cooperate and compete in decision-making processes."},
    {"idx": 38, "neuroscience": "All of these drugs interact in different ways with neurotransmitter and other chemical messenger systems in the brain.",
        "developmental_biology": "All of these drugs interact in different ways with neurotransmitter and other chemical messenger systems in the body."},
    {"idx": 39, "neuroscience": "In many cases, the drugs hijack natural brain systems that have to do with pleasure and reward - psychological processes that are important in eating, drinking, sex and even learning and memory.",
        "developmental_biology": "In many cases, the drugs hijack natural systems that have to do with pleasure and reward - psychological processes that are important in eating, drinking, sex and even learning and pattern memory."},
    {"idx": 40, "neuroscience": "Drugs that act on the brain or the blood supply of the brain can be invaluable - such as those that relieve pain.",
        "developmental_biology": "Drugs that act on the body or the blood supply of the body can be invaluable - such as those that relieve pain."},
    {"idx": 41, "neuroscience": "The average number of neurons in the human brain is 100 billion",
        "developmental_biology": "The average number of cells in the human body is 37 trillion"},
    {"idx": 42, "neuroscience": "Dogs have about 530 million neurons, where as humans have 100 billion neurons.",
        "developmental_biology": "Dogs have trillions of cells, where as humans have 37 trillion cells."},
]

training_translations = {
    'idx': [],
    'document': [],
    'summary': []
}

for t in training_translations_list:
    training_translations['idx'].append(t['idx'])
    training_translations['document'].append(t['neuroscience'])
    training_translations['summary'].append(t['developmental_biology'])

validation_translations_list = [
    {'idx': 1, 'neuroscience': 'Memory is the faculty of the mind by which data or information is encoded, stored, and retrieved when needed.',
        'developmental_biology': 'Pattern memory is the faculty of the body by which data or information is encoded, stored, and retrieved when needed.'},
    {'idx': 2, 'neuroscience': 'This can be related to the neuron.',
        'developmental_biology': 'This can be related to the cell.'},
    {'idx': 3, 'neuroscience': 'The neuron is the smallest unit of the brain.',
        'developmental_biology': 'The cell is the smallest unit of the body.'},
    {'idx': 4, 'neuroscience': 'Learning and memory are usually attributed to changes in neuronal synapses.',
        'developmental_biology': 'Learning and pattern memory are usually attributed to changes in cellular gap junctions.'},
    {'idx': 5, 'neuroscience': 'The synapse is the connection between two neurons.',
        'developmental_biology': 'The gap junction is the connection between two cells.'},
    {'idx': 6, 'neuroscience': 'Neuroscientists have gained much knowledge about the neuronal codes from the studies of plasticity, but most of such research has been focused on simple learning in simple neuronal circuits; it is considerably less clear about the neuronal changes involved in more complex examples of memory',
        'developmental_biology': 'Developmental biologists have gained much knowledge about the cellular codes from the studies of plasticity, but most of such research has been focused on simple learning in simple cellular circuits; it is considerably less clear about the cellular changes involved in more complex examples of pattern memory'},
    {'idx': 7, 'neuroscience': 'Considering that there are several kinds of memory, depending on types of represented knowledge, underlying mechanisms, processes functions and modes of acquisition, it is likely that different brain areas support different memory systems and that they are in mutual relationships in neuronal networks: "components of memory representation are distributed widely across different parts of the brain as mediated by multiple neocortical circuits".',
        'developmental_biology': 'Considering that there are several kinds of pattern memory, depending on types of represented knowledge, underlying mechanisms, processes functions and modes of acquisition, it is likely that different body areas support different pattern memory systems and that they are in mutual relationships in cellular networks: "components of pattern memory representation are distributed widely across different parts of the body as mediated by multiple cellular circuits".'},
    {'idx': 8, 'neuroscience': 'Neuromorphic engineering is a branch of neuroscience that deals with creating functional physical models of neurons for the purposes of useful computation.',
        'developmental_biology': 'Biomorphic engineering is a branch of developmental biology that deals with creating functional physical models of cells for the purposes of useful computation.'},
    {'idx': 9, 'neuroscience': 'Studies tend to indicate small to moderate correlations (averaging around 0.3 to 0.4) between brain volume and IQ.',
     'developmental_biology': 'Studies tend to indicate small to moderate correlations (averaging around 0.3 to 0.4) between body volume and IQ.'},
    {'idx': 10, 'neuroscience': 'During a new learning experience, a set of genes is rapidly expressed in the brain.',
        'developmental_biology': 'During a new learning experience, a set of genes is rapidly expressed in the body.'},
    {'idx': 11, 'neuroscience': 'Sleep affects memory consolidation.',
        'developmental_biology': 'Sleep affects pattern memory consolidation.'},
    {'idx': 12, 'neuroscience': 'During sleep, the neural connections in the brain are strengthened. This enhances the brain\'s abilities to stabilize and retain memories.',
        'developmental_biology': 'During sleep, the cellular connections in the body are strengthened. This enhances the body\'s abilities to stabilize and retain pattern memories.'},
    {'idx': 13, 'neuroscience': 'Encoding of working memory involves the spiking of individual neurons induced by sensory input.',
        'developmental_biology': 'Encoding of working pattern memory involves the spiking of individual cells induced by sensory input.'},
    {'idx': 14, 'neuroscience': 'The spiking of neurons is the process of the firing of neurons.',
        'developmental_biology': 'The spiking of cells is the process of the firing of cells.'},
    {'idx': 15, 'neuroscience': 'Neuroimaging includes the use of various techniques to either directly or indirectly image the structure and function of brains.',
        'developmental_biology': 'Bodyimaging includes the use of various techniques to either directly or indirectly image the structure and function of bodies.'},
    {'idx': 16, 'neuroscience': 'Studies of the brain became more sophisticated with the use of the microscope.',
        'developmental_biology': 'Studies of the body became more sophisticated with the use of the microscope.'},
    {'idx': 17, 'neuroscience': 'The brain is a complex system of neurons.',
        'developmental_biology': 'The body is a complex system of cells.'},
    {'idx': 18, 'neuroscience': 'The brain is composed of many different types of neurons.',
        'developmental_biology': 'The body is composed of many different types of cells.'},
]

validation_translations = {
    'idx': [],
    'document': [],
    'summary': []
}

for t in validation_translations_list:
    validation_translations['idx'].append(t['idx'])
    validation_translations['document'].append(t['neuroscience'])
    validation_translations['summary'].append(t['developmental_biology'])


class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        # torch.tensor(self.labels[idx])
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)


def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts=None, val_labels=None,
                 test_texts=None, test_labels=None):
    """
    Prepare input data for model fine-tuning
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prepare_val = False if val_texts is None or val_labels is None else True
    prepare_test = False if test_texts is None or test_labels is None else True

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True)
        decodings = tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
    test_dataset = tokenize_data(
        test_texts, test_labels) if prepare_test else None

    return train_dataset, val_dataset, test_dataset, tokenizer


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PegasusForConditionalGeneration.from_pretrained(
    model_name).to(torch_device)


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./neuroscience-to-dev-bio-translation'):
    """
    Prepare configurations and base model for fine-tuning
    """

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # if data_args.ignore_pad_token_for_loss:
        if True:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure *
                  100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,           # output directory
        num_train_epochs=20,            # total number of training epochs
        # compute the gradients 8 times and average them before taking a step
        gradient_accumulation_steps=8,
        # batch size per device during training, can increase if memory allows
        per_device_train_batch_size=1,
        # batch size for evaluation, can increase if memory allows
        per_device_eval_batch_size=1,
        fp16=True,                       # lower precision floating points
        #         save_steps=300,                  # number of updates steps before checkpoint saves
        # limit the total amount of checkpoints and deletes the older checkpoints
        save_total_limit=2,
        evaluation_strategy='epoch',     # evaluation strategy to adopt during training
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_strategy='epoch',
        save_strategy='epoch',
        learning_rate=0.0005,
        load_best_model_at_end=True,
        metric_for_best_model='RougeL' if not EARLY_STOP_ON_VAL_LOSS else 'loss',
        predict_with_generate=True,
        skip_memory_metrics=False,
        # push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics if not EARLY_STOP_ON_VAL_LOSS else None,
    )

    return trainer


if __name__ == '__main__':
    train_texts, train_labels = training_translations['document'][:
                                                                  35], training_translations['summary'][:35]
    val_texts, val_labels = validation_translations['document'], validation_translations['summary']
    test_texts, test_labels = training_translations['document'][35:
                                                                ], training_translations['summary'][35:]

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.log(logging.INFO, "STARTING TRAINING")

    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(
        model_name, train_texts, train_labels, val_texts=val_texts, val_labels=val_labels, test_texts=test_texts, test_labels=test_labels)
    trainer = prepare_fine_tuning(
        model_name, tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()

rouge = load_metric('rouge')
max_length = 128
# input_txt = "Testing brains here"
input_ids = tokenizer.batch_encode_plus(
    test_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)["input_ids"].to(torch_device)
predictions = model.generate(
    input_ids,
    max_length=max_length,
    num_beams=10,
    early_stopping=True
)
# predictions_decoded = tokenizer.batch_decode(predictions)
# res = rouge.compute(predictions=predictions_decoded, references=test_labels)

max_length = 128
input_txt = "Memories about specific events in the past."
input_ids = tokenizer(input_txt, return_tensors="pt")[
    "input_ids"].to(torch_device)
print(input_ids)

beam_output = model.generate(
    input_ids,
    max_length=max_length,
    num_beams=10,
    early_stopping=True
)

print("PREDICTION")
print(tokenizer.decode(beam_output[0]))
