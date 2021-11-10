import torch
from torch import nn
from model import WWDModel
from data import Dataset, pad
from functools import partial
import argparse
import textwrap


def train(train_dataset, validation_dataset, test_dataset,
        ambience_csv, path_to_save_model, path_to_wav,
        wake_label, stop_label, num_epochs, validate_every,
        learning_rate_decay_every, learning_rate_decay_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WWDModel().to(device)

    # Create a helper function, so we don't have to type the
    # same arguments multiple times
    def load_dataset(dataset, validation=False):
        return Dataset(dataset, validation=validation,
            ambience_csv_path=ambience_csv, wav_files_path=path_to_wav,
            wake_label=wake_label, stop_label=stop_label)

    loader = torch.utils.data.DataLoader(load_dataset(train_dataset),
        batch_size=4, collate_fn=partial(pad, device), shuffle=True)
    val_loader = torch.utils.data.DataLoader(load_dataset(validation_dataset, validation=True),
        batch_size=1, collate_fn=partial(pad, device), shuffle=True)
    test_loader = torch.utils.data.DataLoader(load_dataset(test_dataset, validation=True),
        batch_size=1, collate_fn=partial(pad, device))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_loss = float("inf")

    try:
        # Wrap the train loop in a try/except, so a keyboard interrupt
        # doesn't immediately exit, but instead goes to the testing loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                spectrograms, labels = data

                optimizer.zero_grad()

                outputs = model(spectrograms)

                loss = criterion(outputs, labels)
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()

                print("epoch %i [ %i / %i ]: running_loss %f\r" % (epoch+1, i,
                    len(loader.dataset) / loader.batch_size, running_loss / (i+1)), end="")
            print()

            if epoch % learning_rate_decay_every == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= learning_rate_decay_rate  # Decay learning rate
                print("lr is now", g["lr"])

            if epoch % validate_every == 0:
                num_wrong = 0
                num_samples = 0
                model.eval()
                with torch.no_grad():
                    val_running_loss = 0.0
                    for i, data in enumerate(val_loader, 0):
                        if i >= 0: # always
                            num_samples += 1

                            spectrograms, labels = data

                            optimizer.zero_grad()

                            outputs = model(spectrograms)

                            test_loss = torch.nn.functional.cross_entropy(
                                    outputs, labels)

                            val_running_loss += test_loss.item()

                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            if labels.item() != torch.argmax(probabilities).item():
                                num_wrong += 1

                    print('val_loss: ', val_running_loss / (i+1), "err. rate: %i/%i" % (num_wrong, num_samples))

                if num_wrong / float(num_samples) + val_running_loss / (i+1) < best_val_loss:
                    best_val_loss = num_wrong / float(num_samples) + val_running_loss / (i+1)
                    torch.save(model, "saved_model.torch")
                    print("saved model to saved_model.torch")

    except KeyboardInterrupt:
        # We assume that a keyboard interupt means stop training, but still carry on to test
        pass

    with torch.no_grad():
        model.eval()
        test_running_loss = 0.0
        num_wrong = 0
        num_samples = 0
        for i, data in enumerate(test_loader, 0):
            if i >= 0: # always
                num_samples += 1
                print("#" * 15)
                print("test", i)
                spectrograms, labels = data

                optimizer.zero_grad()

                outputs = model(spectrograms)

                test_loss = torch.nn.functional.cross_entropy(
                        outputs, labels)

                test_running_loss += test_loss.item()

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                if labels.item() != torch.argmax(probabilities).item():
                    num_wrong += 1
                    print(["wake","stop","pass"][labels.item()], " > (", ["wake","stop","pass"][torch.argmax(probabilities).item()], ")")

                print('test_loss: ', test_running_loss / (i+1))

        print("missed %i from %i" % (num_wrong, num_samples))


class RawHelpFormatter(argparse.HelpFormatter):
    # https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    def _fill_text(self, text, width, indent):
                return "\n".join([textwrap.fill(line, width) for line in\
                    textwrap.indent(textwrap.dedent(text), indent).splitlines()])

if __name__ == "__main__":
    help_text = """
Kronos virtual assistant - Wake word detection trainer

The dataset .csv files need to have the following columns:
   wav_filename wav_filesize transcript

There's an extra ambience csv file, which is more a list of .wav files than a dataset,
but it's used to provide a bit of additional background ambience to the recorded
samples. The ambience recordings are expected to be just background noise in
the room you're planning to use the virtual assistant in.
"""

    parser = argparse.ArgumentParser(
        description=help_text, formatter_class=RawHelpFormatter)
    parser.add_argument("train_dataset",
        help="path to the dataset .csv file for training")
    parser.add_argument("validation_dataset",
        help="path to the dataset .csv file for validation during training")
    parser.add_argument("test_dataset",
        help="path to the dataset .csv file for testing after training")
    parser.add_argument("ambience_csv",
        help="path to the dataset .csv file storing the names of ambience .wav files")
    parser.add_argument("-ps", "--path_to_save_model", default="saved_model.torch",
        help="path to save the trained model at. By default it's a file called "
             "saved_model.torch in the current directory.")
    parser.add_argument("-pw", "--path_to_wav", default="wav",
        help="path to the directory storing the .wav files specified in the "
             "datasets. By default it's 'wav' directory in the current directory.")
    parser.add_argument("-lw", "--label_wake", default="hey kronos",
        help="the text in the wake samples. By default it's 'hey kronos'.")
    parser.add_argument("-ls", "--label_stop", default="stop",
        help="the text in the wake samples. By default it's 'stop'.")
    parser.add_argument("-ne", "--num_epochs", default=1250, type=int,
        help="how many epochs of training to run. By default it's 1250.")
    parser.add_argument("-ve", "--validate_every", default=10, type=int,
        help="how often to validate in epochs. By default it's every 10.")
    parser.add_argument("-lrde", "--learning_rate_decay_every", default=15, type=int,
        help="how often to decay learning rate. By default it's every 15.")
    parser.add_argument("-lrdr", "--learning_rate_decay_rate", default=.99, type=float,
        help="how much to decay learning rate. By default it's .99.")

    parsed_args = parser.parse_args()

    train(train_dataset=parsed_args.train_dataset,
        validation_dataset=parsed_args.validation_dataset,
        test_dataset=parsed_args.test_dataset,
        ambience_csv=parsed_args.ambience_csv,
        path_to_save_model=parsed_args.path_to_save_model,
        path_to_wav=parsed_args.path_to_wav,
        wake_label=parsed_args.label_wake,
        stop_label=parsed_args.label_stop,
        num_epochs=parsed_args.num_epochs,
        validate_every=parsed_args.validate_every,
        learning_rate_decay_every=parsed_args.learning_rate_decay_every,
        learning_rate_decay_rate=parsed_args.learning_rate_decay_rate)
