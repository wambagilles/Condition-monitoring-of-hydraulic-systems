from datetime import datetime
import torch
from tqdm import tqdm as tqdm
from metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter 
from pathlib import Path

def train(model, train_loader, test_loader):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/alnet_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 30

    best_loss = 1_000_000.

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, train_loader, epoch, loss_fn, optimizer, writer)


        running_loss = 0.0
        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        output_list = []
        label_list = []
        with torch.no_grad():
            i=0
            for encoder_input, a_priori_vars, labels in tqdm(test_loader):
                outputs = model(encoder_input.float(), a_priori_vars.float().unsqueeze(1))
                outputs = outputs.flatten(1)
                output_list.append(outputs)
                label_list.append(labels.float())
                loss = loss_fn(outputs, labels.float())
                running_loss += loss
                i+=1

        avg_precision, avg_recall, avg_f1 = compute_metrics(torch.cat(output_list, axis=0), torch.cat(label_list, axis=0))
        avg_loss = running_loss / (i + 1)

        print(f"LOSS train {avg_loss} valid {avg_loss}")
        print(f"Pecision: {avg_precision}, Recall: {avg_f1}, F1: {avg_f1}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_loss },
                        epoch_number + 1)
        writer.add_scalars('Precision', {'Precision' : avg_precision },
                        epoch_number + 1)
        writer.add_scalars('Recall', {'Recall' : avg_recall },
                        epoch_number + 1)
        writer.add_scalars('F1',{'F1' : avg_f1 },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_loss < best_loss:
            best_vloss = avg_loss
            model_path = 'checkpoints/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
    return model



def train_one_epoch(model, training_loader, epoch_index, loss_fn, optimizer, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we want to track batches
    i =0
    for encoder_input, a_priori_vars, labels in tqdm(training_loader):

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(encoder_input.float(), a_priori_vars.float().unsqueeze(1))

        # Compute the loss and its gradients
        outputs = outputs.flatten(1)
        loss = loss_fn(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    i+=1

    return last_loss