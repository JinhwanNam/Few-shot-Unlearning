from utils import *
from loss import *
from prepare import *

import torch
import numpy as np


import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
   
    batch_size = 256
    dataloader = [i for i in range(60000 // batch_size)]

    dataset = args.dataset
    comment = args.comment
    conditional = args.conditional
    normalization = args.normalization
    class_num = args.classes
    percentage = args.percentage
    classes = [i for i in range(class_num)]
    task = args.task

    epochs = 30
    lr = args.learningrate
    b1 = 0.5
    b2 = 0.999
    

    classifier, generator, rotator, unnormalizer, normalizer = prepare_models(dataset, conditional, device, task, class_num=len(classes), normalization=normalization)
    rescale, rescale_mult, totalvar, classifier_mult, diversity_mult, augmentation_mult, meanvar_origin, meanvar_layers = prepare_hyperparameter(args, classifier.model)
    fixed_class, fixed_z = prepare_fixed_value(dataset, device, class_num=len(classes))
    fixed_class = torch.nn.functional.one_hot(fixed_class, num_classes=len(classes)).float()

    

    De = prepare_De(dataset=dataset, task=task, percentage=percentage)
    myfe = FE()
    myfilter = Filter(myfe, De, dataset=dataset)

    classifier.eval()
    with torch.no_grad():
        classifier(myfilter.De)
        meanvar_de = [meanvar_layers[i].mean_var for i in range(len(meanvar_layers))]
        meanvar_de = [(meanvar_de[i][0].detach(), meanvar_de[i][1].detach()) for i in range(len(meanvar_layers))]

    #generator.model.load_state_dict(torch.load("save/old/CIFAR10_main3_NINE_BatchNorm_256_adj03.pyt"))

    #CE_loss = torch.nn.CrossEntropyLoss()    
    CE_loss = softXEnt
    optimizer_G = torch.optim.Adam(generator.model.parameters(), lr=lr, betas=(b1,b2), weight_decay=2e-4)
    #optimizer_G = torch.optim.Adam(generator.model.parameters(), lr=lr, betas=(b1,b2))


    dir_name = f"./data/{dataset}_main3_{task}_{normalization}_{batch_size}_{percentage}" 
    if comment != "nocomment": 
        dir_name = dir_name + f"_{comment}"

    os.makedirs(dir_name, exist_ok=True)

    with open(dir_name + "/configs.txt", 'w') as f:
        f.write(str(args))

    for epoch in range(epochs):
        generator.model.eval()
        classifier.model.eval()

        save_fixed_image(generator, unnormalizer, fixed_z, fixed_class, dir_name, epoch)
        generator.model.train()

        for i in range(len(dataloader)):
             
            z       = torch.tensor(np.random.normal(0, 1, (batch_size, 100))).float().to(device)
            z_class_= torch.cat([torch.tensor(classes), (torch.rand(batch_size - len(classes)) * (max(classes) - min(classes) + 1))]).type(torch.LongTensor).to(device)
            z_class = torch.nn.functional.one_hot(z_class_, num_classes=len(classes)).float()

            x = generator(z, z_class)

            classified1 = classifier(normalizer(rotator(unnormalizer(x))))
            classified2 = classifier(normalizer(rotator(unnormalizer(x))))
            classified3 = classifier(normalizer(rotator(unnormalizer(x))))
            classified4 = classifier(normalizer(rotator(unnormalizer(x))))

            #classified  = classifier(x)

            #meanvar     = meanvar_layers

            x_dr = x[z_class_!=class_num-1]
            x_de = x[z_class_==class_num-1]

            classified_de = classifier(x_de)
            meanvar_de_g  = [meanvar_layers[i].mean_var for i in range(len(meanvar_layers))]
            embeddings_de = classifier.model.embeddings

            classified_dr = classifier(x_dr)
            meanvar_dr_g  = [meanvar_layers[i].mean_var for i in range(len(meanvar_layers))]

            classified = classifier(x)
            meanvar_g  = [meanvar_layers[i].mean_var for i in range(len(meanvar_layers))]
            z_class_dr = z_class[z_class_!=class_num-1][:,:-1]

            mydict = {'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'CROSS7': 2, "BABY": 14}
            z_class_de = torch.full((z_class[z_class_==class_num-1].shape[0],), mydict[task]).to(device)

            bn_loss             = BN_loss               (rescale, meanvar_g, meanvar_origin) 
            bn_loss2            = L2norm                (rescale, meanvar_de, meanvar_de_g)
            bn_loss2           += L2norm                (rescale, meanvar_de, meanvar_dr_g)
            #small_loss          = torch.norm(embeddings_de, 2) * 0.1
            entropy_loss        = CE_loss               (classified_dr, z_class_dr) * classifier_mult * classified_dr.shape[0] / z_class.shape[0]
            entropy_loss2       = CE_loss               (classified_de, z_class_de) * classifier_mult * classified_de.shape[0] / z_class.shape[0]
            diversity_loss      = Diversity_loss        (z, classifier.model.embeddings, z_class_) * diversity_mult
            tv_loss             = Totalvariation_loss   (x) * totalvar
            augmentation_loss   = Augmentation_loss     (classified, [classified1, classified2, classified3, classified4]) * augmentation_mult


            if i == 1:
                my_losses = [entropy_loss, entropy_loss2, augmentation_loss, bn_loss, diversity_loss, tv_loss, bn_loss2]
                check_gradient_of_losses(generator.model, optimizer_G, my_losses)
                print_outputs(classified, classified1, classified2, classified3, classified4)



            g_loss = 0
    
            g_loss += bn_loss
            g_loss += bn_loss2
            g_loss += entropy_loss
            g_loss += entropy_loss2
            g_loss += diversity_loss
            g_loss += tv_loss
            g_loss += augmentation_loss
            #g_loss += small_loss

            optimizer_G.zero_grad() 
            g_loss.backward()
            optimizer_G.step()

                    
    print(args)
    torch.save(generator.model.state_dict(), f"./save/TRUE/{dir_name[7:]}.pyt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="MNIST", help="")
    parser.add_argument("--comment", type=str, required=False, default="nocomment", help="")
    parser.add_argument("--conditional", type=bool, required=False, default=True, help="")
    parser.add_argument("--rescale_mult", type=float, required=False, default=3.0,help="")
    parser.add_argument("--totalvar", type=float, required=False, default=1.0, help="")
    parser.add_argument("--classifier_mult", type=float, required=False, default=3.0, help="")
    parser.add_argument("--diversity_mult", type=float, required=False, default=1.0, help="")
    parser.add_argument("--augmentation_mult", type=float, required=False, default=3.0, help="")
    parser.add_argument("--learningrate", type=float, required=False, default=0.002, help="")
    parser.add_argument("--classes", type=int, required=False, default=11, help="")
    parser.add_argument("--normalization", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="NINE, SEVEN, CROSS7, BABY, ...")
    parser.add_argument("--percentage", type=float, required=False, default=0.03, help="")

    args = parser.parse_args()
    print(args)

    main(args)
