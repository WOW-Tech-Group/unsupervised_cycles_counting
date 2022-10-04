import torch
from torch import save, load
from torch.nn import TripletMarginLoss, MSELoss, Sequential
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from training_utils.sequenced_dataloader import get_dataloaders
from training_utils.triplet_based_encoder import Big_Triplet_Encoder
import os


def train_model(frames_and_flow_path, force_reset,
                epochs_nb=30, latent_space_size=32, batch_size=32, print_losses=False):

    model_path = os.path.join(frames_and_flow_path, "embedding_model.pth")
    model = Big_Triplet_Encoder(latent_space_size=latent_space_size, input_channels=8).cuda()

    if os.path.isfile(model_path) and not force_reset :
        model.load_state_dict(load(model_path))
        print("A model already exists")
        return model

    dataloader = get_dataloaders(frames_and_flow_path, batch_size=batch_size, train_test_ratio=1)
    while (len(dataloader.dataset)-1)%batch_size == 0 :
        batch_size +=1
        dataloader = get_dataloaders(frames_and_flow_path, batch_size=batch_size, train_test_ratio=1)

    criterion = TripletMarginLoss(margin=0.1)
    optimizer = Adam(model.parameters(),
                    lr=1e-3,
                    weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    scheduler2 = MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)

    print('No pre-existing model.'
          '\nTraining starts')

    for epoch in range(epochs_nb) :
        total_epoch_loss = 0
        for batch in dataloader :
            outputs = []
            for _, img in batch.items() : # iterates through current, next, next_next
                img = img.cuda()
                bottleneck = model.forward(img)
                outputs.append(bottleneck)

            loss = criterion(*outputs)
            total_epoch_loss += float(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(float(loss))
            scheduler2.step()

        total_epoch_loss /= len(dataloader)
        if print_losses: print("\t" + str(total_epoch_loss), epoch+1, "/", epochs_nb)

    save(model.state_dict(), model_path)

    print("Training complete.")
    torch.cuda.empty_cache()
    return model