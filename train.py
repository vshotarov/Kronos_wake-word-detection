import torch
from torch import nn
from model import WWDModel
from data import Dataset, pad
from functools import partial


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = WWDModel().to(device)
    model.train()
    num_params = sum([p.numel() for p in model.parameters()])
    loader = torch.utils.data.DataLoader(Dataset("wake_word_train.csv"), batch_size=4,
            collate_fn=partial(pad, device), shuffle=True)
    val_loader = torch.utils.data.DataLoader(Dataset("wake_word_test.csv", validation=True),
            batch_size=1, collate_fn=partial(pad, device), shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset("wake_word_test.csv", validation=True),
            batch_size=1, collate_fn=partial(pad, device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_loss = 1000

    try:
        for epoch in range(1250):
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

            if epoch % 15 == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= .99
                    lr = g["lr"]
                print("lr is now", lr)

            if epoch % 10 == 0:
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

if __name__ == "__main__":
    train()
