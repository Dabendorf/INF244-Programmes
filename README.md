# Programmes useful for INF244 course at UiB
This repository includes a lot of algorithms being useful for the UiB course [``INF244 Graph-based Inference, Networks and Coding Theory``](https://www.uib.no/en/course/INF244)


It includes simulation of different channels, repetitional codes, Convolutional encoders, Viterbi decoders, Turbo encoders, LDPC decoder and Polar codes

# Channels
## AWGN
Parameters: 0.5 = n_0 (sigma), 0.3 = energy level<br>
``python ChannelAWGN.py uib.png 0.5 0.3 png``<br>
``python ChannelAWGN.py Mondrian.csv 0.5 0.3 csv``<br>

## BSC
Parameters: 0.1 = prob<br>
``python ChannelBSC.py uib.png 0.1 png``<br>
``python ChannelBSC.py Mondrian.csv 0.1 csv``<br>

## BEC
Parameters: 0.1 = prob<br>
``python ChannelBEC.py uib.png 0.1 png``<br>
``python ChannelBEC.py Mondrian.csv 0.1 csv``<br>

# Repetitional code
Parameters: 0.3 = prob for BEC/BSC, 0.6 = E_b/N_0 for AWGN<br>
``python Repcode.py Mondrian.csv 5 bsc 0.3``<br>
``python Repcode.py Mondrian.csv 5 bec 0.3``<br>
``python Repcode.py Mondrian.csv 5 awgn 0.6``<br>

# Convolutional encoder
Parameters: k, n = Size of the Matrix, matrix = Matrix seperated by S going row by row, seq = Sequence to encode<br>
``python ConvFinal.py k n matrix seq``<br>
``python ConvFinal.py 1 2 111S101 100100``<br>
``python ConvFinal.py 1 2 1111S1011 1011011``<br>

## Old programme, sometimes not working, but for checking
``python ConvOld.py k n b m``<br>
``python ConvOld.py 1 2 1111S1011 1011011``<br>

# Turbo encoder
``python TurboEncoder.py 1S1101 csv Sequence.csv Permutation.csv``<br>
``python TurboEncoder.py 1S1101 str 10110 4S5S3S1S2``<br>

# Viterbi decoder
Parameter: k, n = Size of the Matrix, channel = {awgn, bsc, bec}, seq = Sequence to encode, matrix = Matrix seperated by S going row by row<br>
``python Viterbi.py k n channel seq matrix``<br>
``python Viterbi.py 1 2 bsc 111S101 000111011001110000``<br>
``python Viterbi.py 1 2 bsc 111S001 000111011001110000``<br>
``python Viterbi.py 2 3 bsc 111S11S101S1S01S11 111000011100000100``<br>
``python Viterbi.py 1 2 bec 111S101 2220111222``<br>
``python Viterbi.py 1 2 awgn 111S101 -0.3S0.0S-0.9S0.2S0.9S0.6S1.1S-0.7S1.4S1.1``<br>

# LDPC Decoder
## AWGN 
G can actually be a dummy if no mhat decoding necessary<br>
That will make max likelihood useless<br>
``python LDPCdecoderAWGN.py systematic paritycheck sequence N0 debug_mode``<br>
``python LDPCdecoderAWGN.py G H r N0 debug_mode``<br>
``python LDPCdecoderAWGN.py G.csv H4.csv -0.2S0.3S-1.2S0.5S-0.8S-0.6S1.1 4 True``<br>

## BSC
``python LDPCdecoderBSC.py systematic paritycheck sequence probability debug_mode``<br>
``python LDPCdecoderBSC.py G H r prob debug_mode``<br>
``python LDPCdecoderBSC.py G.csv H_bec_lecturenotes.csv 011111 0.25 True``<br>

## BEC
``python LDPCdecoderBEC.py systematic paritycheck sequence probability debug_mode``<br>
``python LDPCdecoderBEC.py G H r prob debug_mode``<br>
``python LDPCdecoderBEC.py G.csv H_bec_lecturenotes.csv 020212 0.25 True``<br>

# Polar codes
Parameter: Amount of iterations<br>
``python3 PolarCodes.py 3``<br>

## Polarise
Calculate the probabilities of erasure probability in all channels<br>
Input is initial erasure probability<br>
``echo 0.3 | ./polarise``<br>
``echo 0.3 | ./polarise | ./polarise``<br>

## PolariseExp
Prints calculation path<br>
``echo p | ./polariseExp``<br>
``echo p | ./polariseExp | ./polariseExp``<br>
