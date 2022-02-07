from utils.helpers import *


if __name__ == '__main__':
    fasta_sequences = SeqIO.parse(open('F:/BioInformatic/Dataset/TFPM vs TFPNM/TFPM_independent_dataset.txt'), 'fasta')
    fasta_sequences1 = SeqIO.parse(open('F:/BioInformatic/Dataset/TFPM vs TFPNM/TFPM_mouse.txt'), 'fasta')
    fasta_sequences2 = SeqIO.parse(open('F:/BioInformatic/Dataset/TFPM vs TFPNM/TFPM_training_dataset_unused.txt'), 'fasta')
    fasta_sequences3 = SeqIO.parse(open('F:/BioInformatic/Dataset/TFPM vs TFPNM/TFPM_training_dataset_used.txt'), 'fasta')
    fasta_sequences4 = SeqIO.parse(open('F:/BioInformatic/Dataset/TFPM vs TFPNM/TFPNM_independent_dataset.txt'), 'fasta')
    fasta_sequences5 = SeqIO.parse(open('F:/BioInformatic/Dataset/TFPM vs TFPNM/TFPNM_training_dataset.txt'), 'fasta')
    i = 0
    for fasta in fasta_sequences:
        i += 1
    print(i)

    i = 0
    for fasta in fasta_sequences1:
        i += 1
    print(i)

    i = 0
    for fasta in fasta_sequences2:
        i += 1
    print(i)

    i = 0
    for fasta in fasta_sequences3:
        i += 1
    print(i)

    i = 0
    for fasta in fasta_sequences4:
        i += 1
    print(i)

    i = 0
    for fasta in fasta_sequences5:
        i += 1
    print(i)

    fasta_sequences6 = SeqIO.parse(open(
        'F:/BioInformatic/Dataset/TF vs non-TF/non-TF_training_dataset.txt'),
                                   'fasta')
    fasta_sequences7 = SeqIO.parse(open(
        'F:/BioInformatic/Dataset/TF vs non-TF/TF_training_dataset.txt'),
                                   'fasta')
    i = 0
    for fasta in fasta_sequences6:
        i += 1
    print(i)

    i = 0
    for fasta in fasta_sequences7:
        i += 1
    print(i)