ð»:
ýÒ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018á»5
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:d*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:d*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:d*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:d*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
d*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:d*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:d*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:d*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:d*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
d*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
h
StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
StateVar
a
StateVar/Read/ReadVariableOpReadVariableOpStateVar*
_output_shapes
:*
dtype0	
l

StateVar_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
StateVar_1
e
StateVar_1/Read/ReadVariableOpReadVariableOp
StateVar_1*
_output_shapes
:*
dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:d*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:d*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:d*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:d*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
d*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:*
dtype0

block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv4/kernel

'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:*
dtype0

block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv3/kernel

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:*
dtype0

block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv4/kernel

'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:*
dtype0

block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv4/kernel

'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
æË
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* Ë
valueËBË BË
Ï
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer_with_weights-15
layer-22
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&
signatures*
§
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator*
§
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator*
* 
È
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
È
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op*

G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
È
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op*
È
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op*

_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
È
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op*
È
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias
 v_jit_compiled_convolution_op*
È
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias
 _jit_compiled_convolution_op*
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
! _jit_compiled_convolution_op*
Ñ
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§kernel
	¨bias
!©_jit_compiled_convolution_op*
Ñ
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses
°kernel
	±bias
!²_jit_compiled_convolution_op*

³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses* 
Ñ
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses
¿kernel
	Àbias
!Á_jit_compiled_convolution_op*
Ñ
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses
Èkernel
	Ébias
!Ê_jit_compiled_convolution_op*
Ñ
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses
Ñkernel
	Òbias
!Ó_jit_compiled_convolution_op*
Ñ
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses
Úkernel
	Ûbias
!Ü_jit_compiled_convolution_op*

Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses* 

ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç__call__
+è&call_and_return_all_conditional_losses* 
®
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses
ïkernel
	ðbias*
à
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses
	÷axis

øgamma
	ùbeta
úmoving_mean
ûmoving_variance*

ü	variables
ýtrainable_variables
þregularization_losses
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ô
;0
<1
D2
E3
S4
T5
\6
]7
k8
l9
t10
u11
}12
~13
14
15
16
17
18
19
§20
¨21
°22
±23
¿24
À25
È26
É27
Ñ28
Ò29
Ú30
Û31
ï32
ð33
ø34
ù35
ú36
û37
38
39*
4
ï0
ð1
ø2
ù3
4
5*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
Í
	iter
beta_1
beta_2

decay
learning_rate	ïmö	ðm÷	ømø	ùmù	mú	mû	ïvü	ðvý	øvþ	ùvÿ	v	v*

serving_default* 
* 
* 
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

¢trace_0
£trace_1* 

¤trace_0
¥trace_1* 

¦
_generator*
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

¬trace_0
­trace_1* 

®trace_0
¯trace_1* 

°
_generator*

;0
<1*
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

¶trace_0* 

·trace_0* 
c]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

D0
E1*
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

½trace_0* 

¾trace_0* 
c]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

Ätrace_0* 

Åtrace_0* 

S0
T1*
* 
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

Ëtrace_0* 

Ìtrace_0* 
c]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

\0
]1*
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

Òtrace_0* 

Ótrace_0* 
c]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

Ùtrace_0* 

Útrace_0* 

k0
l1*
* 
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

àtrace_0* 

átrace_0* 
c]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

t0
u1*
* 
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

çtrace_0* 

ètrace_0* 
c]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

}0
~1*
* 
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

îtrace_0* 

ïtrace_0* 
c]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

õtrace_0* 

ötrace_0* 
c]
VARIABLE_VALUEblock3_conv4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock3_conv4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ütrace_0* 

ýtrace_0* 

0
1*
* 
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
c]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
c]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

§0
¨1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses*

trace_0* 

trace_0* 
d^
VARIABLE_VALUEblock4_conv3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock4_conv3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

°0
±1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses*

trace_0* 

trace_0* 
d^
VARIABLE_VALUEblock4_conv4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock4_conv4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses* 

trace_0* 

 trace_0* 

¿0
À1*
* 
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses*

¦trace_0* 

§trace_0* 
d^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

È0
É1*
* 
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses*

­trace_0* 

®trace_0* 
d^
VARIABLE_VALUEblock5_conv2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ñ0
Ò1*
* 
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*

´trace_0* 

µtrace_0* 
d^
VARIABLE_VALUEblock5_conv3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ú0
Û1*
* 
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses*

»trace_0* 

¼trace_0* 
d^
VARIABLE_VALUEblock5_conv4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEblock5_conv4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses* 

Âtrace_0* 

Ãtrace_0* 
* 
* 
* 

Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses* 

Étrace_0* 

Êtrace_0* 

ï0
ð1*

ï0
ð1*
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses*

Ðtrace_0* 

Ñtrace_0* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
ø0
ù1
ú2
û3*

ø0
ù1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses*

×trace_0
Øtrace_1* 

Ùtrace_0
Útrace_1* 
* 
ic
VARIABLE_VALUEbatch_normalization/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEbatch_normalization/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEbatch_normalization/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#batch_normalization/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
ü	variables
ýtrainable_variables
þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

àtrace_0* 

átrace_0* 

0
1*

0
1*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

çtrace_0* 

ètrace_0* 
_Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_1/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1
D2
E3
S4
T5
\6
]7
k8
l9
t10
u11
}12
~13
14
15
16
17
18
19
§20
¨21
°22
±23
¿24
À25
È26
É27
Ñ28
Ò29
Ú30
Û31
ú32
û33*
â
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28*

é0
ê1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ë
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

ì
_state_var*

;0
<1*
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

S0
T1*
* 
* 
* 
* 
* 
* 

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

k0
l1*
* 
* 
* 
* 
* 
* 

t0
u1*
* 
* 
* 
* 
* 
* 

}0
~1*
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 

§0
¨1*
* 
* 
* 
* 
* 
* 

°0
±1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

¿0
À1*
* 
* 
* 
* 
* 
* 

È0
É1*
* 
* 
* 
* 
* 
* 

Ñ0
Ò1*
* 
* 
* 
* 
* 
* 

Ú0
Û1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ú0
û1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
í	variables
î	keras_api

ïtotal

ðcount*
M
ñ	variables
ò	keras_api

ótotal

ôcount
õ
_fn_kwargs*
nh
VARIABLE_VALUE
StateVar_1Jlayer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEStateVarJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

ï0
ð1*

í	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ó0
ô1*

ñ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
z
VARIABLE_VALUEAdam/dense/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/mRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/mQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/vRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/vQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

!serving_default_random_zoom_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
Á	
StatefulPartitionedCallStatefulPartitionedCall!serving_default_random_zoom_inputblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense/kernel
dense/biasbatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gammadense_1/kerneldense_1/bias*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_16369
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¤
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpStateVar_1/Read/ReadVariableOpStateVar/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*L
TinE
C2A			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_19217
³
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense/kernel
dense/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate
StateVar_1StateVartotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_19416ÌÅ2
ú
¤
,__inference_block3_conv3_layer_call_fn_18634

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_14349x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs


G__inference_block4_conv4_layer_call_and_return_conditional_losses_18755

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ô
9loop_body_stateful_uniform_full_int_pfor_while_cond_15122n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1n
jloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_15122___redundant_placeholder0
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_15122___redundant_placeholder1
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_15122___redundant_placeholder2
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_15122___redundant_placeholder3;
7loop_body_stateful_uniform_full_int_pfor_while_identity

3loop_body/stateful_uniform_full_int/pfor/while/LessLess:loop_body_stateful_uniform_full_int_pfor_while_placeholderjloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity7loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
9


Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_17935~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1{
wloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityE
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3y
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	~
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ï
:loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderEloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPackBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderUloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1Pack>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0Wloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0Sloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÉ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcastMloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Eloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
[loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderJloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ó
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :«
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ®
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ñ
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: °
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2Identity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: Û
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identitykloop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"ð
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slicewloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

b
F__inference_block2_pool_layer_call_and_return_conditional_losses_18585

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
K
/__inference_random_contrast_layer_call_fn_17777

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_14232j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

©
J__inference_random_contrast_layer_call_and_return_conditional_losses_18485

inputsI
;loop_body_stateful_uniform_full_int_rngreadandskip_resource:	
identity¢2loop_body/stateful_uniform_full_int/RngReadAndSkip¢=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:s
)loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(loop_body/stateful_uniform_full_int/ProdProd2loop_body/stateful_uniform_full_int/shape:output:02loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*loop_body/stateful_uniform_full_int/Cast_1Cast1loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
7loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/stateful_uniform_full_int/strided_sliceStridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+loop_body/stateful_uniform_full_int/BitcastBitcast:loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3loop_body/stateful_uniform_full_int/strided_slice_1StridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-loop_body/stateful_uniform_full_int/Bitcast_1Bitcast<loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Ã
#loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV22loop_body/stateful_uniform_full_int/shape:output:06loop_body/stateful_uniform_full_int/Bitcast_1:output:04loop_body/stateful_uniform_full_int/Bitcast:output:00loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 
loop_body/stackPack,loop_body/stateful_uniform_full_int:output:0loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
loop_body/strided_slice_1StridedSliceloop_body/stack:output:0(loop_body/strided_slice_1/stack:output:0*loop_body/strided_slice_1/stack_1:output:0*loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskk
(loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?k
&loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *33³?¥
?loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"loop_body/strided_slice_1:output:0* 
_output_shapes
::
?loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21loop_body/stateless_random_uniform/shape:output:0Eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&loop_body/stateless_random_uniform/subSub/loop_body/stateless_random_uniform/max:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&loop_body/stateless_random_uniform/mulMulDloop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"loop_body/stateless_random_uniformAddV2*loop_body/stateless_random_uniform/mul:z:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
loop_body/adjust_contrastAdjustContrastv2loop_body/GatherV2:output:0&loop_body/stateless_random_uniform:z:0*$
_output_shapes
:
"loop_body/adjust_contrast/IdentityIdentity"loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:f
!loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C²
loop_body/clip_by_value/MinimumMinimum+loop_body/adjust_contrast/Identity:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:^
loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
loop_body/clip_by_valueMaximum#loop_body/clip_by_value/Minimum:z:0"loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:\
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Kloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Tloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Sloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReserve\loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Ploop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhileSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0Yloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:03^loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *T
bodyLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_17870*T
condLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_17869*#
output_shapes
: : : : : : : : 
?loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ©
Xloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ´
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Bloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
=loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2Oloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Kloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:å
Cloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Mloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReserveUloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌx
6loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Iloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
6loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhileLloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0Rloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0?loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0Lloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *M
bodyERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_17935*M
condERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_17934*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ{
8loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¢
Qloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2?loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3Zloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Aloop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2Qloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Mloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
Eloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Oloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿË
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReserveWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌz
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Kloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ®
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhileNloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0Tloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *O
bodyGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_18002*O
condGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_18001*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ}
:loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¤
Sloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3\loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
<loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ü
6loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Eloop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ­
6loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReserveMloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐp
.loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ}
;loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë
.loop_body/stateful_uniform_full_int/pfor/whileStatelessWhileDloop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0Jloop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:07loop_body/stateful_uniform_full_int/pfor/Const:output:0?loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:02loop_body/stateful_uniform_full_int/shape:output:00loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *E
body=R;
9loop_body_stateful_uniform_full_int_pfor_while_body_18059*E
cond=R;
9loop_body_stateful_uniform_full_int_pfor_while_cond_18058*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: s
0loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Iloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿø
;loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV27loop_body/stateful_uniform_full_int/pfor/while:output:3Rloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:09loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0d
loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:f
$loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :¹
loop_body/stack/pfor/ones_likeFill=loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0-loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:u
"loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¢
loop_body/stack/pfor/ReshapeReshape'loop_body/stack/pfor/ones_like:output:0+loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
$loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
loop_body/stack/pfor/Reshape_1Reshapepfor/Reshape:output:0-loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:b
 loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
loop_body/stack/pfor/concatConcatV2'loop_body/stack/pfor/Reshape_1:output:0%loop_body/stack/pfor/Reshape:output:0)loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:e
#loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : £
loop_body/stack/pfor/ExpandDims
ExpandDimsloop_body/zeros_like:output:0,loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:£
loop_body/stack/pfor/TileTile(loop_body/stack/pfor/ExpandDims:output:0$loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
loop_body/stack/pfor/stackPackDloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:0"loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axisx
.loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: l
*loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
%loop_body/strided_slice_1/pfor/concatConcatV27loop_body/strided_slice_1/pfor/concat/values_0:output:0(loop_body/strided_slice_1/stack:output:03loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: n
,loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_1ConcatV29loop_body/strided_slice_1/pfor/concat_1/values_0:output:0*loop_body/strided_slice_1/stack_1:output:05loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:n
,loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_2ConcatV29loop_body/strided_slice_1/pfor/concat_2/values_0:output:0*loop_body/strided_slice_1/stack_2:output:05loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
+loop_body/strided_slice_1/pfor/StridedSliceStridedSlice#loop_body/stack/pfor/stack:output:0.loop_body/strided_slice_1/pfor/concat:output:00loop_body/strided_slice_1/pfor/concat_1:output:00loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask¢
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlicepfor/Reshape:output:0aloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask«
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReserveiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ­
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Tloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReservekloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¨
]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhile`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0floop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:04loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *a
bodyYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_18159*a
condYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_18158*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¶
eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   è
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ¸
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlicepfor/Reshape:output:0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask§
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserveeloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¤
Yloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Æ

Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhile\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0bloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:01loop_body/stateless_random_uniform/shape:output:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *]
bodyURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_18228*]
condURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_18227*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 
Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ´
aloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÔ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0r
0loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :t
2loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : s
1loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ò
/loop_body/stateless_random_uniform/mul/pfor/addAddV2;loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0:loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: Ï
3loop_body/stateless_random_uniform/mul/pfor/MaximumMaximum3loop_body/stateless_random_uniform/mul/pfor/add:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ½
1loop_body/stateless_random_uniform/mul/pfor/ShapeShape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:Ë
/loop_body/stateless_random_uniform/mul/pfor/subSub7loop_body/stateless_random_uniform/mul/pfor/Maximum:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
9loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ü
3loop_body/stateless_random_uniform/mul/pfor/ReshapeReshape3loop_body/stateless_random_uniform/mul/pfor/sub:z:0Bloop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
6loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ú
0loop_body/stateless_random_uniform/mul/pfor/TileTile?loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0<loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
?loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Hloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
;loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masky
7loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
2loop_body/stateless_random_uniform/mul/pfor/concatConcatV2Bloop_body/stateless_random_uniform/mul/pfor/strided_slice:output:09loop_body/stateless_random_uniform/mul/pfor/Tile:output:0Dloop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0@loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
5loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0;loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
/loop_body/stateless_random_uniform/mul/pfor/MulMul>loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :p
.loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : o
-loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Æ
+loop_body/stateless_random_uniform/pfor/addAddV27loop_body/stateless_random_uniform/pfor/Rank_1:output:06loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: Ã
/loop_body/stateless_random_uniform/pfor/MaximumMaximum/loop_body/stateless_random_uniform/pfor/add:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
-loop_body/stateless_random_uniform/pfor/ShapeShape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:¿
+loop_body/stateless_random_uniform/pfor/subSub3loop_body/stateless_random_uniform/pfor/Maximum:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
5loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/loop_body/stateless_random_uniform/pfor/ReshapeReshape/loop_body/stateless_random_uniform/pfor/sub:z:0>loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,loop_body/stateless_random_uniform/pfor/TileTile;loop_body/stateless_random_uniform/pfor/Tile/input:output:08loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5loop_body/stateless_random_uniform/pfor/strided_sliceStridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Dloop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Floop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.loop_body/stateless_random_uniform/pfor/concatConcatV2>loop_body/stateless_random_uniform/pfor/strided_slice:output:05loop_body/stateless_random_uniform/pfor/Tile:output:0@loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0<loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ø
1loop_body/stateless_random_uniform/pfor/Reshape_1Reshape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:07loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
-loop_body/stateless_random_uniform/pfor/AddV2AddV2:loop_body/stateless_random_uniform/pfor/Reshape_1:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : î
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
,loop_body/adjust_contrast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0;loop_body/adjust_contrast/pfor/strided_slice/stack:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserveCloop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:05loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
$loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
7loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ï
$loop_body/adjust_contrast/pfor/whileStatelessWhile:loop_body/adjust_contrast/pfor/while/loop_counter:output:0@loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0-loop_body/adjust_contrast/pfor/Const:output:05loop_body/adjust_contrast/pfor/TensorArrayV2:handle:05loop_body/adjust_contrast/pfor/strided_slice:output:0)loop_body/GatherV2/pfor/GatherV2:output:01loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *;
body3R1
/loop_body_adjust_contrast_pfor_while_body_18377*;
cond3R1
/loop_body_adjust_contrast_pfor_while_cond_18376*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿi
&loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
?loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         Ú
1loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2-loop_body/adjust_contrast/pfor/while:output:3Hloop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0´
0loop_body/adjust_contrast/Identity/pfor/IdentityIdentity:loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/clip_by_value/Minimum/pfor/addAddV24loop_body/clip_by_value/Minimum/pfor/Rank_1:output:03loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/clip_by_value/Minimum/pfor/MaximumMaximum,loop_body/clip_by_value/Minimum/pfor/add:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/clip_by_value/Minimum/pfor/ShapeShape9loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:¶
(loop_body/clip_by_value/Minimum/pfor/subSub0loop_body/clip_by_value/Minimum/pfor/Maximum:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/clip_by_value/Minimum/pfor/ReshapeReshape,loop_body/clip_by_value/Minimum/pfor/sub:z:0;loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/clip_by_value/Minimum/pfor/TileTile8loop_body/clip_by_value/Minimum/pfor/Tile/input:output:05loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Aloop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/clip_by_value/Minimum/pfor/concatConcatV2;loop_body/clip_by_value/Minimum/pfor/strided_slice:output:02loop_body/clip_by_value/Minimum/pfor/Tile:output:0=loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:09loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:æ
.loop_body/clip_by_value/Minimum/pfor/Reshape_1Reshape9loop_body/adjust_contrast/Identity/pfor/Identity:output:04loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
,loop_body/clip_by_value/Minimum/pfor/MinimumMinimum7loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : d
"loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¥
 loop_body/clip_by_value/pfor/addAddV2,loop_body/clip_by_value/pfor/Rank_1:output:0+loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: ¢
$loop_body/clip_by_value/pfor/MaximumMaximum$loop_body/clip_by_value/pfor/add:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
"loop_body/clip_by_value/pfor/ShapeShape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:
 loop_body/clip_by_value/pfor/subSub(loop_body/clip_by_value/pfor/Maximum:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: t
*loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¯
$loop_body/clip_by_value/pfor/ReshapeReshape$loop_body/clip_by_value/pfor/sub:z:03loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:­
!loop_body/clip_by_value/pfor/TileTile0loop_body/clip_by_value/pfor/Tile/input:output:0-loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
*loop_body/clip_by_value/pfor/strided_sliceStridedSlice+loop_body/clip_by_value/pfor/Shape:output:09loop_body/clip_by_value/pfor/strided_slice/stack:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
,loop_body/clip_by_value/pfor/strided_slice_1StridedSlice+loop_body/clip_by_value/pfor/Shape:output:0;loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
#loop_body/clip_by_value/pfor/concatConcatV23loop_body/clip_by_value/pfor/strided_slice:output:0*loop_body/clip_by_value/pfor/Tile:output:05loop_body/clip_by_value/pfor/strided_slice_1:output:01loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
&loop_body/clip_by_value/pfor/Reshape_1Reshape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0,loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
&loop_body/clip_by_value/pfor/Maximum_1Maximum/loop_body/clip_by_value/pfor/Reshape_1:output:0"loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity*loop_body/clip_by_value/pfor/Maximum_1:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp3^loop_body/stateful_uniform_full_int/RngReadAndSkip>^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2h
2loop_body/stateful_uniform_full_int/RngReadAndSkip2loop_body/stateful_uniform_full_int/RngReadAndSkip2~
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
n
¿
F__inference_random_zoom_layer_call_and_return_conditional_losses_15664

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *333?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àD
í
Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_17008£
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter©
¤random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderZ
Vrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1 
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0³
®random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityW
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1W
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2W
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice±
¬random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¥
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
:  
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPackTrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholdergrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¢
`random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1PackPrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0irandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¯
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¯
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSlice®random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0erandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0grandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0grandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskí
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcast_random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ç
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsYrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0`random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Î
mrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemVrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :©
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderYrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ô
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterYrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Ò
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityRrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: §
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identity¤random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ô
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2IdentityRrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: ÿ
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identity}random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityZrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"³
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"³
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"³
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"º
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slicerandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0"à
¬random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice®random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
û
¡
,__inference_block1_conv2_layer_call_fn_18514

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_14262y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸:
¸

Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_15066
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityG
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3}
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :õ
<loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPackDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1Pack@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0Yloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0Uloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÍ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcastOloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
]loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderLloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ù
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :³
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ²
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: ÷
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ´
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2IdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: ß
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identitymloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"ø
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0" 
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


G__inference_block2_conv1_layer_call_and_return_conditional_losses_18555

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿH@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@
 
_user_specified_nameinputs

b
F__inference_block4_pool_layer_call_and_return_conditional_losses_14118

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
F__inference_random_zoom_layer_call_and_return_conditional_losses_14226

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
F__inference_block4_pool_layer_call_and_return_conditional_losses_18765

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
F__inference_block5_pool_layer_call_and_return_conditional_losses_14130

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
»
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_17934~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_17934___redundant_placeholder0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity
¤
;loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLessBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderzloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: ­
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

b
F__inference_block2_pool_layer_call_and_return_conditional_losses_14094

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block2_conv2_layer_call_and_return_conditional_losses_18575

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿH: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
¹
	
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_18158§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¾
¹loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_18158___redundant_placeholder0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
õ
Oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: Õ
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitySloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


N__inference_batch_normalization_layer_call_and_return_conditional_losses_18931

inputs*
cast_readvariableop_resource:d,
cast_1_readvariableop_resource:d,
cast_2_readvariableop_resource:d,
cast_3_readvariableop_resource:d
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:d*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:dm
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:dm
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ë	
ó
@__inference_dense_layer_call_and_return_conditional_losses_18885

inputs2
matmul_readvariableop_resource:
d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
¤
,__inference_block4_conv3_layer_call_fn_18724

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_14418x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
É

/__inference_random_contrast_layer_call_fn_17784

inputs
unknown:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_15549y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_18227
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_18227___redundant_placeholder0¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_18227___redundant_placeholder1¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_18227___redundant_placeholder2¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_18227___redundant_placeholder3S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
å
Kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: Í
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityOloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
ê
Ý

#__inference_signature_wrapper_16369
random_zoom_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:
d

unknown_32:d

unknown_33:d

unknown_34:d

unknown_35:d

unknown_36:d

unknown_37:d

unknown_38:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallrandom_zoom_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_14073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_zoom_input

b
F__inference_random_zoom_layer_call_and_return_conditional_losses_17670

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_1_layer_call_and_return_conditional_losses_14562

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
L
Ú
?random_contrast_loop_body_adjust_contrast_pfor_while_body_17383z
vrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counter
|random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterationsD
@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderF
Brandom_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1w
srandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slice_0y
urandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2_0
random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0A
=random_contrast_loop_body_adjust_contrast_pfor_while_identityC
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_1C
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_2C
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_3u
qrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slicew
srandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2
random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2|
:random_contrast/loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :é
8random_contrast/loop_body/adjust_contrast/pfor/while/addAddV2@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderCrandom_contrast/loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Hrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stackPack@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderSrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1Pack<random_contrast/loop_body/adjust_contrast/pfor/while/add:z:0Urandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¯
Brandom_contrast/loop_body/adjust_contrast/pfor/while/strided_sliceStridedSliceurandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2_0Qrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0Srandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0Srandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask~
<random_contrast/loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :í
:random_contrast/loop_body/adjust_contrast/pfor/while/add_1AddV2@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderErandom_contrast/loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPack@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderUrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Nrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1Pack>random_contrast/loop_body/adjust_contrast/pfor/while/add_1:z:0Wrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¹
Drandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicerandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0Srandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0Urandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0Urandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_mask
Erandom_contrast/loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2Krandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice:output:0Mrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:
Crandom_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?random_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDimsNrandom_contrast/loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0Lrandom_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:þ
Yrandom_contrast/loop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBrandom_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderHrandom_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ~
<random_contrast/loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :í
:random_contrast/loop_body/adjust_contrast/pfor/while/add_2AddV2@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderErandom_contrast/loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ~
<random_contrast/loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :£
:random_contrast/loop_body/adjust_contrast/pfor/while/add_3AddV2vrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counterErandom_contrast/loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ª
=random_contrast/loop_body/adjust_contrast/pfor/while/IdentityIdentity>random_contrast/loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: ê
?random_contrast/loop_body/adjust_contrast/pfor/while/Identity_1Identity|random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¬
?random_contrast/loop_body/adjust_contrast/pfor/while/Identity_2Identity>random_contrast/loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ×
?random_contrast/loop_body/adjust_contrast/pfor/while/Identity_3Identityirandom_contrast/loop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
=random_contrast_loop_body_adjust_contrast_pfor_while_identityFrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity:output:0"
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_1Hrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity_1:output:0"
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_2Hrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity_2:output:0"
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_3Hrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity_3:output:0"è
qrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slicesrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slice_0"
random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0"ì
srandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2urandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
®Î
×$
E__inference_sequential_layer_call_and_return_conditional_losses_17654

inputsB
4random_zoom_stateful_uniform_rngreadandskip_resource:	Y
Krandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	E
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block4_conv4_conv2d_readvariableop_resource:;
,block4_conv4_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	G
+block5_conv4_conv2d_readvariableop_resource:;
,block5_conv4_biasadd_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
d3
%dense_biasadd_readvariableop_resource:dI
;batch_normalization_assignmovingavg_readvariableop_resource:dK
=batch_normalization_assignmovingavg_1_readvariableop_resource:d>
0batch_normalization_cast_readvariableop_resource:d@
2batch_normalization_cast_1_readvariableop_resource:d8
&dense_1_matmul_readvariableop_resource:d5
'dense_1_biasadd_readvariableop_resource:
identity¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢'batch_normalization/Cast/ReadVariableOp¢)batch_normalization/Cast_1/ReadVariableOp¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block4_conv4/BiasAdd/ReadVariableOp¢"block4_conv4/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp¢#block5_conv4/BiasAdd/ReadVariableOp¢"block5_conv4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢Brandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip¢Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while¢+random_zoom/stateful_uniform/RngReadAndSkipG
random_zoom/ShapeShapeinputs*
T0*
_output_shapes
:i
random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom/strided_sliceStridedSlicerandom_zoom/Shape:output:0(random_zoom/strided_slice/stack:output:0*random_zoom/strided_slice/stack_1:output:0*random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
!random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
#random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿm
#random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom/strided_slice_1StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_1/stack:output:0,random_zoom/strided_slice_1/stack_1:output:0,random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
random_zoom/CastCast$random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: t
!random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿv
#random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom/strided_slice_2StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_2/stack:output:0,random_zoom/strided_slice_2/stack_1:output:0,random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
random_zoom/Cast_1Cast$random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: f
$random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :«
"random_zoom/stateful_uniform/shapePack"random_zoom/strided_slice:output:0-random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:e
 random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *333?e
 random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
"random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¤
!random_zoom/stateful_uniform/ProdProd+random_zoom/stateful_uniform/shape:output:0+random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: e
#random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
#random_zoom/stateful_uniform/Cast_1Cast*random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: æ
+random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkip4random_zoom_stateful_uniform_rngreadandskip_resource,random_zoom/stateful_uniform/Cast/x:output:0'random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:z
0random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:è
*random_zoom/stateful_uniform/strided_sliceStridedSlice3random_zoom/stateful_uniform/RngReadAndSkip:value:09random_zoom/stateful_uniform/strided_slice/stack:output:0;random_zoom/stateful_uniform/strided_slice/stack_1:output:0;random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
$random_zoom/stateful_uniform/BitcastBitcast3random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0|
2random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
,random_zoom/stateful_uniform/strided_slice_1StridedSlice3random_zoom/stateful_uniform/RngReadAndSkip:value:0;random_zoom/stateful_uniform/strided_slice_1/stack:output:0=random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0=random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
&random_zoom/stateful_uniform/Bitcast_1Bitcast5random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0{
9random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ë
5random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2+random_zoom/stateful_uniform/shape:output:0/random_zoom/stateful_uniform/Bitcast_1:output:0-random_zoom/stateful_uniform/Bitcast:output:0Brandom_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 random_zoom/stateful_uniform/subSub)random_zoom/stateful_uniform/max:output:0)random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: ¿
 random_zoom/stateful_uniform/mulMul>random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0$random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
random_zoom/stateful_uniformAddV2$random_zoom/stateful_uniform/mul:z:0)random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
random_zoom/concatConcatV2 random_zoom/stateful_uniform:z:0 random_zoom/stateful_uniform:z:0 random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
random_zoom/zoom_matrix/ShapeShaperandom_zoom/concat:output:0*
T0*
_output_shapes
:u
+random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%random_zoom/zoom_matrix/strided_sliceStridedSlice&random_zoom/zoom_matrix/Shape:output:04random_zoom/zoom_matrix/strided_slice/stack:output:06random_zoom/zoom_matrix/strided_slice/stack_1:output:06random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom/zoom_matrix/subSubrandom_zoom/Cast_1:y:0&random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: f
!random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
random_zoom/zoom_matrix/truedivRealDivrandom_zoom/zoom_matrix/sub:z:0*random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
-random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_1StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_1/stack:output:08random_zoom/zoom_matrix/strided_slice_1/stack_1:output:08random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskd
random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
random_zoom/zoom_matrix/sub_1Sub(random_zoom/zoom_matrix/sub_1/x:output:00random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
random_zoom/zoom_matrix/mulMul#random_zoom/zoom_matrix/truediv:z:0!random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom/zoom_matrix/sub_2Subrandom_zoom/Cast:y:0(random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: h
#random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
!random_zoom/zoom_matrix/truediv_1RealDiv!random_zoom/zoom_matrix/sub_2:z:0,random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
-random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_2StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_2/stack:output:08random_zoom/zoom_matrix/strided_slice_2/stack_1:output:08random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskd
random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
random_zoom/zoom_matrix/sub_3Sub(random_zoom/zoom_matrix/sub_3/x:output:00random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
random_zoom/zoom_matrix/mul_1Mul%random_zoom/zoom_matrix/truediv_1:z:0!random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_3StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_3/stack:output:08random_zoom/zoom_matrix/strided_slice_3/stack_1:output:08random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskh
&random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :»
$random_zoom/zoom_matrix/zeros/packedPack.random_zoom/zoom_matrix/strided_slice:output:0/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
#random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
random_zoom/zoom_matrix/zerosFill-random_zoom/zoom_matrix/zeros/packed:output:0,random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¿
&random_zoom/zoom_matrix/zeros_1/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
random_zoom/zoom_matrix/zeros_1Fill/random_zoom/zoom_matrix/zeros_1/packed:output:0.random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_4StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_4/stack:output:08random_zoom/zoom_matrix/strided_slice_4/stack_1:output:08random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskj
(random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¿
&random_zoom/zoom_matrix/zeros_2/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
random_zoom/zoom_matrix/zeros_2Fill/random_zoom/zoom_matrix/zeros_2/packed:output:0.random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :·
random_zoom/zoom_matrix/concatConcatV20random_zoom/zoom_matrix/strided_slice_3:output:0&random_zoom/zoom_matrix/zeros:output:0random_zoom/zoom_matrix/mul:z:0(random_zoom/zoom_matrix/zeros_1:output:00random_zoom/zoom_matrix/strided_slice_4:output:0!random_zoom/zoom_matrix/mul_1:z:0(random_zoom/zoom_matrix/zeros_2:output:0,random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
random_zoom/transform/ShapeShapeinputs*
T0*
_output_shapes
:s
)random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
#random_zoom/transform/strided_sliceStridedSlice$random_zoom/transform/Shape:output:02random_zoom/transform/strided_slice/stack:output:04random_zoom/transform/strided_slice/stack_1:output:04random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:e
 random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Í
0random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputs'random_zoom/zoom_matrix/concat:output:0,random_zoom/transform/strided_slice:output:0)random_zoom/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
random_contrast/ShapeShapeErandom_zoom/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:m
#random_contrast/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_contrast/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_contrast/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
random_contrast/strided_sliceStridedSlicerandom_contrast/Shape:output:0,random_contrast/strided_slice/stack:output:0.random_contrast/strided_slice/stack_1:output:0.random_contrast/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
random_contrast/Rank/packedPack&random_contrast/strided_slice:output:0*
N*
T0*
_output_shapes
:V
random_contrast/RankConst*
_output_shapes
: *
dtype0*
value	B :]
random_contrast/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
random_contrast/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¥
random_contrast/rangeRange$random_contrast/range/start:output:0random_contrast/Rank:output:0$random_contrast/range/delta:output:0*
_output_shapes
:w
random_contrast/Max/inputPack&random_contrast/strided_slice:output:0*
N*
T0*
_output_shapes
:
random_contrast/MaxMax"random_contrast/Max/input:output:0random_contrast/range:output:0*
T0*
_output_shapes
: x
6random_contrast/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ½
0random_contrast/loop_body/PlaceholderWithDefaultPlaceholderWithDefault?random_contrast/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: 
random_contrast/loop_body/ShapeShapeErandom_zoom/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:w
-random_contrast/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/random_contrast/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_contrast/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'random_contrast/loop_body/strided_sliceStridedSlice(random_contrast/loop_body/Shape:output:06random_contrast/loop_body/strided_slice/stack:output:08random_contrast/loop_body/strided_slice/stack_1:output:08random_contrast/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#random_contrast/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :­
!random_contrast/loop_body/GreaterGreater0random_contrast/loop_body/strided_slice:output:0,random_contrast/loop_body/Greater/y:output:0*
T0*
_output_shapes
: f
$random_contrast/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : à
"random_contrast/loop_body/SelectV2SelectV2%random_contrast/loop_body/Greater:z:09random_contrast/loop_body/PlaceholderWithDefault:output:0-random_contrast/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: i
'random_contrast/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¦
"random_contrast/loop_body/GatherV2GatherV2Erandom_zoom/transform/ImageProjectiveTransformV3:transformed_images:0+random_contrast/loop_body/SelectV2:output:00random_contrast/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:
9random_contrast/loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
9random_contrast/loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
8random_contrast/loop_body/stateful_uniform_full_int/ProdProdBrandom_contrast/loop_body/stateful_uniform_full_int/shape:output:0Brandom_contrast/loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: |
:random_contrast/loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :µ
:random_contrast/loop_body/stateful_uniform_full_int/Cast_1CastArandom_contrast/loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Â
Brandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipKrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourceCrandom_contrast/loop_body/stateful_uniform_full_int/Cast/x:output:0>random_contrast/loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Grandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Irandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Irandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
Arandom_contrast/loop_body/stateful_uniform_full_int/strided_sliceStridedSliceJrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Prandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÃ
;random_contrast/loop_body/stateful_uniform_full_int/BitcastBitcastJrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Irandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Krandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Krandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
Crandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1StridedSliceJrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ç
=random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1BitcastLrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0y
7random_contrast/loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
3random_contrast/loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV2Brandom_contrast/loop_body/stateful_uniform_full_int/shape:output:0Frandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1:output:0Drandom_contrast/loop_body/stateful_uniform_full_int/Bitcast:output:0@random_contrast/loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	n
$random_contrast/loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Æ
random_contrast/loop_body/stackPack<random_contrast/loop_body/stateful_uniform_full_int:output:0-random_contrast/loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:
/random_contrast/loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1random_contrast/loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1random_contrast/loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)random_contrast/loop_body/strided_slice_1StridedSlice(random_contrast/loop_body/stack:output:08random_contrast/loop_body/strided_slice_1/stack:output:0:random_contrast/loop_body/strided_slice_1/stack_1:output:0:random_contrast/loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask{
8random_contrast/loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB {
6random_contrast/loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?{
6random_contrast/loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *33³?Å
Orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter2random_contrast/loop_body/strided_slice_1:output:0* 
_output_shapes
::
Orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Î
Krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Arandom_contrast/loop_body/stateless_random_uniform/shape:output:0Urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Xrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: à
6random_contrast/loop_body/stateless_random_uniform/subSub?random_contrast/loop_body/stateless_random_uniform/max:output:0?random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ð
6random_contrast/loop_body/stateless_random_uniform/mulMulTrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0:random_contrast/loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: Ù
2random_contrast/loop_body/stateless_random_uniformAddV2:random_contrast/loop_body/stateless_random_uniform/mul:z:0?random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: È
)random_contrast/loop_body/adjust_contrastAdjustContrastv2+random_contrast/loop_body/GatherV2:output:06random_contrast/loop_body/stateless_random_uniform:z:0*$
_output_shapes
:¡
2random_contrast/loop_body/adjust_contrast/IdentityIdentity2random_contrast/loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:v
1random_contrast/loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Câ
/random_contrast/loop_body/clip_by_value/MinimumMinimum;random_contrast/loop_body/adjust_contrast/Identity:output:0:random_contrast/loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:n
)random_contrast/loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
'random_contrast/loop_body/clip_by_valueMaximum3random_contrast/loop_body/clip_by_value/Minimum:z:02random_contrast/loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:l
"random_contrast/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
random_contrast/pfor/ReshapeReshaperandom_contrast/Max:output:0+random_contrast/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:b
 random_contrast/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 random_contrast/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¼
random_contrast/pfor/rangeRange)random_contrast/pfor/range/start:output:0random_contrast/Max:output:0)random_contrast/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
[random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: §
]random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:§
]random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Urandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0drandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0frandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0frandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
crandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Urandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReservelrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : «
`random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Zrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü	
Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhilecrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0irandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0Krandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourceCrandom_contrast/loop_body/stateful_uniform_full_int/Cast/x:output:0>random_contrast/loop_body/stateful_uniform_full_int/Cast_1:y:0C^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *d
body\RZ
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_16876*d
cond\RZ
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_16875*#
output_shapes
: : : : : : : : 
Orandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¹
hrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
Zrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3qrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0 
Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Mrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2_random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0Prandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack:output:0[random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Orandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2arandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0]random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Orandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2arandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0]random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:µ
Srandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSlicecrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Trandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0]random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0_random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0_random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask§
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReserveerandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Frandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¤
Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¬
Frandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhile\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0brandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0\random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *]
bodyURS
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_16941*]
condURS
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_16940*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Hrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ²
arandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ø
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3jrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Orandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2arandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0]random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Qrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2crandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0_random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¤
Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Qrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2crandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0_random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:½
Urandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSlicecrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask 
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0_random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0arandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0arandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask©
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿû
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReservegrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Hrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¦
[random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Urandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
Hrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhile^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0drandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0^random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *_
bodyWRU
Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_17008*_
condWRU
Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_17007*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Jrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ´
crandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
Urandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0Urandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÝ
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReserve]random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0Orandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
>random_contrast/loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Qrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Krandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 

>random_contrast/loop_body/stateful_uniform_full_int/pfor/whileStatelessWhileTrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0Zrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:0Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/Const:output:0Orandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0Orandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:0Brandom_contrast/loop_body/stateful_uniform_full_int/shape:output:0@random_contrast/loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *U
bodyMRK
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_body_17065*U
condMRK
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_17064*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
@random_contrast/loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ª
Yrandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¸
Krandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV2Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/while:output:3brandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:0Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0t
*random_contrast/loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
Drandom_contrast/loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:v
4random_contrast/loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :é
.random_contrast/loop_body/stack/pfor/ones_likeFillMrandom_contrast/loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0=random_contrast/loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:
2random_contrast/loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÒ
,random_contrast/loop_body/stack/pfor/ReshapeReshape7random_contrast/loop_body/stack/pfor/ones_like:output:0;random_contrast/loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
4random_contrast/loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÄ
.random_contrast/loop_body/stack/pfor/Reshape_1Reshape%random_contrast/pfor/Reshape:output:0=random_contrast/loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:r
0random_contrast/loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+random_contrast/loop_body/stack/pfor/concatConcatV27random_contrast/loop_body/stack/pfor/Reshape_1:output:05random_contrast/loop_body/stack/pfor/Reshape:output:09random_contrast/loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:u
3random_contrast/loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ó
/random_contrast/loop_body/stack/pfor/ExpandDims
ExpandDims-random_contrast/loop_body/zeros_like:output:0<random_contrast/loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:Ó
)random_contrast/loop_body/stack/pfor/TileTile8random_contrast/loop_body/stack/pfor/ExpandDims:output:04random_contrast/loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*random_contrast/loop_body/stack/pfor/stackPackTrandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:02random_contrast/loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axis
>random_contrast/loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:random_contrast/loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5random_contrast/loop_body/strided_slice_1/pfor/concatConcatV2Grandom_contrast/loop_body/strided_slice_1/pfor/concat/values_0:output:08random_contrast/loop_body/strided_slice_1/stack:output:0Crandom_contrast/loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@random_contrast/loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<random_contrast/loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7random_contrast/loop_body/strided_slice_1/pfor/concat_1ConcatV2Irandom_contrast/loop_body/strided_slice_1/pfor/concat_1/values_0:output:0:random_contrast/loop_body/strided_slice_1/stack_1:output:0Erandom_contrast/loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@random_contrast/loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<random_contrast/loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7random_contrast/loop_body/strided_slice_1/pfor/concat_2ConcatV2Irandom_contrast/loop_body/strided_slice_1/pfor/concat_2/values_0:output:0:random_contrast/loop_body/strided_slice_1/stack_2:output:0Erandom_contrast/loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:½
;random_contrast/loop_body/strided_slice_1/pfor/StridedSliceStridedSlice3random_contrast/loop_body/stack/pfor/stack:output:0>random_contrast/loop_body/strided_slice_1/pfor/concat:output:0@random_contrast/loop_body/strided_slice_1/pfor/concat_1:output:0@random_contrast/loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask²
hrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ´
jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:´
jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0qrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0srandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0srandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask»
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ±
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReserveyrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ½
rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿµ
drandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReserve{random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¸
mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¨

Zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhileprandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0vrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0Drandom_contrast/loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *q
bodyiRg
erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_17165*q
condiRg
erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_17164*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Æ
urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¨
grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3~random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 È
wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ­
irandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0®
drandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: °
frandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:°
frandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask·
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¥
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserveurandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Vrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ´
irandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¥
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
Vrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhilelrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:0Arandom_contrast/loop_body/stateless_random_uniform/shape:output:0Xrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *m
bodyeRc
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_17234*m
condeRc
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_17233*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 
Xrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Ä
qrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
@random_contrast/loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
Brandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
Arandom_contrast/loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
?random_contrast/loop_body/stateless_random_uniform/mul/pfor/addAddV2Krandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0Jrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ÿ
Crandom_contrast/loop_body/stateless_random_uniform/mul/pfor/MaximumMaximumCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/add:z:0Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: Ý
Arandom_contrast/loop_body/stateless_random_uniform/mul/pfor/ShapeShapelrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:û
?random_contrast/loop_body/stateless_random_uniform/mul/pfor/subSubGrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Maximum:z:0Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Crandom_contrast/loop_body/stateless_random_uniform/mul/pfor/ReshapeReshapeCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/sub:z:0Rrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Frandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
@random_contrast/loop_body/stateless_random_uniform/mul/pfor/TileTileOrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0Lrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Orandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Qrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Qrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSliceJrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Xrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0Zrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0Zrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Qrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Srandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Srandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ÿ
Krandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSliceJrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Zrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0\random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0\random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
Grandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
Brandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concatConcatV2Rrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice:output:0Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile:output:0Trandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0Prandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¹
Erandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshapelrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0Krandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?random_contrast/loop_body/stateless_random_uniform/mul/pfor/MulMulNrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0:random_contrast/loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
<random_contrast/loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
>random_contrast/loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
=random_contrast/loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ö
;random_contrast/loop_body/stateless_random_uniform/pfor/addAddV2Grandom_contrast/loop_body/stateless_random_uniform/pfor/Rank_1:output:0Frandom_contrast/loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: ó
?random_contrast/loop_body/stateless_random_uniform/pfor/MaximumMaximum?random_contrast/loop_body/stateless_random_uniform/pfor/add:z:0Erandom_contrast/loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: °
=random_contrast/loop_body/stateless_random_uniform/pfor/ShapeShapeCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:ï
;random_contrast/loop_body/stateless_random_uniform/pfor/subSubCrandom_contrast/loop_body/stateless_random_uniform/pfor/Maximum:z:0Erandom_contrast/loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
Erandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?random_contrast/loop_body/stateless_random_uniform/pfor/ReshapeReshape?random_contrast/loop_body/stateless_random_uniform/pfor/sub:z:0Nrandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Brandom_contrast/loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:þ
<random_contrast/loop_body/stateless_random_uniform/pfor/TileTileKrandom_contrast/loop_body/stateless_random_uniform/pfor/Tile/input:output:0Hrandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Krandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
Erandom_contrast/loop_body/stateless_random_uniform/pfor/strided_sliceStridedSliceFrandom_contrast/loop_body/stateless_random_uniform/pfor/Shape:output:0Trandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0Vrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0Vrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Mrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Orandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Orandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
Grandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSliceFrandom_contrast/loop_body/stateless_random_uniform/pfor/Shape:output:0Vrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0Xrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0Xrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
Crandom_contrast/loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
>random_contrast/loop_body/stateless_random_uniform/pfor/concatConcatV2Nrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice:output:0Erandom_contrast/loop_body/stateless_random_uniform/pfor/Tile:output:0Prandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0Lrandom_contrast/loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Arandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape_1ReshapeCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Mul:z:0Grandom_contrast/loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_contrast/loop_body/stateless_random_uniform/pfor/AddV2AddV2Jrandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape_1:output:0?random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,random_contrast/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : o
-random_contrast/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ä
+random_contrast/loop_body/SelectV2/pfor/addAddV25random_contrast/loop_body/SelectV2/pfor/Rank:output:06random_contrast/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: p
.random_contrast/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :p
.random_contrast/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : q
/random_contrast/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ê
-random_contrast/loop_body/SelectV2/pfor/add_1AddV27random_contrast/loop_body/SelectV2/pfor/Rank_2:output:08random_contrast/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Å
/random_contrast/loop_body/SelectV2/pfor/MaximumMaximum7random_contrast/loop_body/SelectV2/pfor/Rank_1:output:0/random_contrast/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Å
1random_contrast/loop_body/SelectV2/pfor/Maximum_1Maximum1random_contrast/loop_body/SelectV2/pfor/add_1:z:03random_contrast/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: 
-random_contrast/loop_body/SelectV2/pfor/ShapeShape#random_contrast/pfor/range:output:0*
T0*
_output_shapes
:Ã
+random_contrast/loop_body/SelectV2/pfor/subSub5random_contrast/loop_body/SelectV2/pfor/Maximum_1:z:07random_contrast/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
5random_contrast/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/random_contrast/loop_body/SelectV2/pfor/ReshapeReshape/random_contrast/loop_body/SelectV2/pfor/sub:z:0>random_contrast/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2random_contrast/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,random_contrast/loop_body/SelectV2/pfor/TileTile;random_contrast/loop_body/SelectV2/pfor/Tile/input:output:08random_contrast/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;random_contrast/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_contrast/loop_body/SelectV2/pfor/strided_sliceStridedSlice6random_contrast/loop_body/SelectV2/pfor/Shape:output:0Drandom_contrast/loop_body/SelectV2/pfor/strided_slice/stack:output:0Frandom_contrast/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Frandom_contrast/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_contrast/loop_body/SelectV2/pfor/strided_slice_1StridedSlice6random_contrast/loop_body/SelectV2/pfor/Shape:output:0Frandom_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Hrandom_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Hrandom_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3random_contrast/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.random_contrast/loop_body/SelectV2/pfor/concatConcatV2>random_contrast/loop_body/SelectV2/pfor/strided_slice:output:05random_contrast/loop_body/SelectV2/pfor/Tile:output:0@random_contrast/loop_body/SelectV2/pfor/strided_slice_1:output:0<random_contrast/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:È
1random_contrast/loop_body/SelectV2/pfor/Reshape_1Reshape#random_contrast/pfor/range:output:07random_contrast/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
0random_contrast/loop_body/SelectV2/pfor/SelectV2SelectV2%random_contrast/loop_body/Greater:z:0:random_contrast/loop_body/SelectV2/pfor/Reshape_1:output:0-random_contrast/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5random_contrast/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ý
0random_contrast/loop_body/GatherV2/pfor/GatherV2GatherV2Erandom_zoom/transform/ImageProjectiveTransformV3:transformed_images:09random_contrast/loop_body/SelectV2/pfor/SelectV2:output:0>random_contrast/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Brandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Drandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Drandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
<random_contrast/loop_body/adjust_contrast/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0Krandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack:output:0Mrandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0Mrandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Jrandom_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¿
<random_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserveSrandom_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:0Erandom_contrast/loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
4random_contrast/loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Grandom_contrast/loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Arandom_contrast/loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
4random_contrast/loop_body/adjust_contrast/pfor/whileStatelessWhileJrandom_contrast/loop_body/adjust_contrast/pfor/while/loop_counter:output:0Prandom_contrast/loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0=random_contrast/loop_body/adjust_contrast/pfor/Const:output:0Erandom_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2:handle:0Erandom_contrast/loop_body/adjust_contrast/pfor/strided_slice:output:09random_contrast/loop_body/GatherV2/pfor/GatherV2:output:0Arandom_contrast/loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *K
bodyCRA
?random_contrast_loop_body_adjust_contrast_pfor_while_body_17383*K
condCRA
?random_contrast_loop_body_adjust_contrast_pfor_while_cond_17382*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿy
6random_contrast/loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¨
Orandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         
Arandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2=random_contrast/loop_body/adjust_contrast/pfor/while:output:3Xrandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0?random_contrast/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0Ô
@random_contrast/loop_body/adjust_contrast/Identity/pfor/IdentityIdentityJrandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_contrast/loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;random_contrast/loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : |
:random_contrast/loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :í
8random_contrast/loop_body/clip_by_value/Minimum/pfor/addAddV2Drandom_contrast/loop_body/clip_by_value/Minimum/pfor/Rank_1:output:0Crandom_contrast/loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: ê
<random_contrast/loop_body/clip_by_value/Minimum/pfor/MaximumMaximum<random_contrast/loop_body/clip_by_value/Minimum/pfor/add:z:0Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: ³
:random_contrast/loop_body/clip_by_value/Minimum/pfor/ShapeShapeIrandom_contrast/loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:æ
8random_contrast/loop_body/clip_by_value/Minimum/pfor/subSub@random_contrast/loop_body/clip_by_value/Minimum/pfor/Maximum:z:0Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_contrast/loop_body/clip_by_value/Minimum/pfor/ReshapeReshape<random_contrast/loop_body/clip_by_value/Minimum/pfor/sub:z:0Krandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_contrast/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_contrast/loop_body/clip_by_value/Minimum/pfor/TileTileHrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Tile/input:output:0Erandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSliceCrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Shape:output:0Qrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0Srandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0Srandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSliceCrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Shape:output:0Srandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0Urandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0Urandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_contrast/loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_contrast/loop_body/clip_by_value/Minimum/pfor/concatConcatV2Krandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice:output:0Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Tile:output:0Mrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:0Irandom_contrast/loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
>random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape_1ReshapeIrandom_contrast/loop_body/adjust_contrast/Identity/pfor/Identity:output:0Drandom_contrast/loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<random_contrast/loop_body/clip_by_value/Minimum/pfor/MinimumMinimumGrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0:random_contrast/loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
1random_contrast/loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :u
3random_contrast/loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : t
2random_contrast/loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Õ
0random_contrast/loop_body/clip_by_value/pfor/addAddV2<random_contrast/loop_body/clip_by_value/pfor/Rank_1:output:0;random_contrast/loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: Ò
4random_contrast/loop_body/clip_by_value/pfor/MaximumMaximum4random_contrast/loop_body/clip_by_value/pfor/add:z:0:random_contrast/loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: ¢
2random_contrast/loop_body/clip_by_value/pfor/ShapeShape@random_contrast/loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:Î
0random_contrast/loop_body/clip_by_value/pfor/subSub8random_contrast/loop_body/clip_by_value/pfor/Maximum:z:0:random_contrast/loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
:random_contrast/loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ß
4random_contrast/loop_body/clip_by_value/pfor/ReshapeReshape4random_contrast/loop_body/clip_by_value/pfor/sub:z:0Crandom_contrast/loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
7random_contrast/loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ý
1random_contrast/loop_body/clip_by_value/pfor/TileTile@random_contrast/loop_body/clip_by_value/pfor/Tile/input:output:0=random_contrast/loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: 
@random_contrast/loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Brandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Brandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
:random_contrast/loop_body/clip_by_value/pfor/strided_sliceStridedSlice;random_contrast/loop_body/clip_by_value/pfor/Shape:output:0Irandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack:output:0Krandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0Krandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Brandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Drandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Drandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
<random_contrast/loop_body/clip_by_value/pfor/strided_slice_1StridedSlice;random_contrast/loop_body/clip_by_value/pfor/Shape:output:0Krandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0Mrandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0Mrandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskz
8random_contrast/loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ø
3random_contrast/loop_body/clip_by_value/pfor/concatConcatV2Crandom_contrast/loop_body/clip_by_value/pfor/strided_slice:output:0:random_contrast/loop_body/clip_by_value/pfor/Tile:output:0Erandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1:output:0Arandom_contrast/loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ý
6random_contrast/loop_body/clip_by_value/pfor/Reshape_1Reshape@random_contrast/loop_body/clip_by_value/Minimum/pfor/Minimum:z:0<random_contrast/loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
6random_contrast/loop_body/clip_by_value/pfor/Maximum_1Maximum?random_contrast/loop_body/clip_by_value/pfor/Reshape_1:output:02random_contrast/loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0é
block1_conv1/Conv2DConv2D:random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:0*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ë
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¦
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHt
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Î
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¦
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHt
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	­
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  
flatten/ReshapeReshapeblock5_pool/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ·
 batch_normalization/moments/meanMeandense/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:d¿
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ú
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<ª
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:d*
dtype0½
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:d´
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:dü
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:d*
dtype0Ã
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:dº
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:d
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:d*
dtype0
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:d*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:­
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:dx
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:d©
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:d
#batch_normalization/batchnorm/mul_1Muldense/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¤
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:d§
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d®
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_1/MatMulMatMulactivation/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOpC^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipN^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while,^random_zoom/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2
Brandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipBrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip2
Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileMrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while2Z
+random_zoom/stateful_uniform/RngReadAndSkip+random_zoom/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®>
å
/loop_body_adjust_contrast_pfor_while_body_18377Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1W
Sloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0Y
Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0h
dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_01
-loop_body_adjust_contrast_pfor_while_identity3
/loop_body_adjust_contrast_pfor_while_identity_13
/loop_body_adjust_contrast_pfor_while_identity_23
/loop_body_adjust_contrast_pfor_while_identity_3U
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceW
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2f
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2l
*loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(loop_body/adjust_contrast/pfor/while/addAddV20loop_body_adjust_contrast_pfor_while_placeholder3loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: |
:loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : å
8loop_body/adjust_contrast/pfor/while/strided_slice/stackPack0loop_body_adjust_contrast_pfor_while_placeholderCloop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:~
<loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : å
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_1Pack,loop_body/adjust_contrast/pfor/while/add:z:0Eloop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
2loop_body/adjust_contrast/pfor/while/strided_sliceStridedSliceUloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0Aloop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskn
,loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_1AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ~
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : é
:loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPack0loop_body_adjust_contrast_pfor_while_placeholderEloop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
>loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ë
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1Pack.loop_body/adjust_contrast/pfor/while/add_1:z:0Gloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
4loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicedloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0Cloop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_maskë
5loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2;loop_body/adjust_contrast/pfor/while/strided_slice:output:0=loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:u
3loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : î
/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDims>loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0<loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:¾
Iloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2loop_body_adjust_contrast_pfor_while_placeholder_10loop_body_adjust_contrast_pfor_while_placeholder8loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒn
,loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_2AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: n
,loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*loop_body/adjust_contrast/pfor/while/add_3AddV2Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter5loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity.loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: º
/loop_body/adjust_contrast/pfor/while/Identity_1Identity\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
/loop_body/adjust_contrast/pfor/while/Identity_2Identity.loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ·
/loop_body/adjust_contrast/pfor/while/Identity_3IdentityYloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_18loop_body/adjust_contrast/pfor/while/Identity_1:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_28loop_body/adjust_contrast/pfor/while/Identity_2:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_38loop_body/adjust_contrast/pfor/while/Identity_3:output:0"¨
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceSloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0"Ê
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0"¬
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
çw
¿
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_17234¿
ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterÅ
Àrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsf
brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderh
drandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1¼
·random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0Ó
Îrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0×
Òrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0
random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape_0­
¨random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0c
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identitye
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1e
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2e
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3º
µrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceÑ
Ìrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2Õ
Ðrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1
random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape«
¦random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ï
Zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdererandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: ®
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : û
jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackbrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderurandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:°
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : û
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1Pack^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:½
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
drandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSliceÎrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0srandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask 
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ó
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: °
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ÿ
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackbrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderwrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:²
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1Pack`random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¿
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
frandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSliceÒrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÂ
orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape_0mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0¨random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: §
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : þ
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimsxrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:
{random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemdrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderjrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ 
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Ó
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
:  
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :¬
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_countergrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: î
_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentity`random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: Ñ
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1IdentityÀrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: ð
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2Identity`random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identityrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "Ë
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityhrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"Ï
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"Ï
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"Ï
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"¦
random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shaperandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape_0"Ô
¦random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg¨random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"ò
µrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice·random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"¨
Ðrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1Òrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0" 
Ìrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2Îrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


G__inference_block4_conv2_layer_call_and_return_conditional_losses_18715

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ð
Õ
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_18001
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_18001___redundant_placeholder0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity
¬
=loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLessDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: ±
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityAloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

b
F__inference_block3_pool_layer_call_and_return_conditional_losses_18675

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
Ù

*__inference_sequential_layer_call_fn_16454

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:
d

unknown_32:d

unknown_33:d

unknown_34:d

unknown_35:d

unknown_36:d

unknown_37:d

unknown_38:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_14569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸:
¸

Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_18002
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityG
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3}
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :õ
<loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPackDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1Pack@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0Yloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0Uloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÍ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcastOloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
]loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderLloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ù
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :³
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ²
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: ÷
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ´
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2IdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: ß
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identitymloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"ø
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0" 
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á

'__inference_dense_1_layer_call_fn_18984

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14562o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


G__inference_block3_conv3_layer_call_and_return_conditional_losses_14349

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
a

Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_body_17065
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counter
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterationsN
Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderP
Lrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice_0©
¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0©
¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0~
zrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shape_0|
xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg_0K
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityM
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_1M
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_2M
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_3
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice§
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2§
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2|
xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shapez
vrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Brandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/addAddV2Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderMrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPackJrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1PackFrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add:z:0_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¥
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
Lrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSlice¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0[random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderOrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ·
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPackJrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Xrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¹
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1PackHrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0arandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:§
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Nrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSlice¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÙ
^random_contrast/loop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2zrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shape_0Urandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
Mrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ê
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsgrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
crandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemLrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderRrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderOrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counterOrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ¾
Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentityHrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: 
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identityrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
: À
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_2IdentityHrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: ë
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identitysrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityPrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_1Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_2Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_3Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0"ò
vrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_algxrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg_0"
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slicerandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"ö
xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shapezrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shape_0"Ì
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"Ì
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
û
£
,__inference_block2_conv1_layer_call_fn_18544

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_14280y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿH@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@
 
_user_specified_nameinputs
Ð
Õ
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_15065
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_15065___redundant_placeholder0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity
¬
=loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLessDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: ±
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityAloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


G__inference_block5_conv1_layer_call_and_return_conditional_losses_18785

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
ú
¤
,__inference_block5_conv4_layer_call_fn_18834

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_14504x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs


G__inference_block5_conv3_layer_call_and_return_conditional_losses_14487

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
­
á
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_16875­
¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter³
®random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterations]
Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_
[random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1­
¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceÄ
¿random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_16875___redundant_placeholder0Ä
¿random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_16875___redundant_placeholder1Ä
¿random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_16875___redundant_placeholder2Z
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity

Rrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessYrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: Û
Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityVrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "¹
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:

b
F__inference_block1_pool_layer_call_and_return_conditional_losses_18535

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block5_conv4_layer_call_and_return_conditional_losses_14504

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

ä

*__inference_sequential_layer_call_fn_14652
random_zoom_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:
d

unknown_32:d

unknown_33:d

unknown_34:d

unknown_35:d

unknown_36:d

unknown_37:d

unknown_38:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallrandom_zoom_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*J
_read_only_resource_inputs,
*(	
 !"#$%&'(*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_14569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_zoom_input
Ì
ÿ 
E__inference_sequential_layer_call_and_return_conditional_losses_16696

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block4_conv4_conv2d_readvariableop_resource:;
,block4_conv4_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	G
+block5_conv4_conv2d_readvariableop_resource:;
,block5_conv4_biasadd_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
d3
%dense_biasadd_readvariableop_resource:d>
0batch_normalization_cast_readvariableop_resource:d@
2batch_normalization_cast_1_readvariableop_resource:d@
2batch_normalization_cast_2_readvariableop_resource:d@
2batch_normalization_cast_3_readvariableop_resource:d8
&dense_1_matmul_readvariableop_resource:d5
'dense_1_biasadd_readvariableop_resource:
identity¢'batch_normalization/Cast/ReadVariableOp¢)batch_normalization/Cast_1/ReadVariableOp¢)batch_normalization/Cast_2/ReadVariableOp¢)batch_normalization/Cast_3/ReadVariableOp¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block4_conv4/BiasAdd/ReadVariableOp¢"block4_conv4/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp¢#block5_conv4/BiasAdd/ReadVariableOp¢"block5_conv4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ë
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¦
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHt
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Î
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¦
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHt
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides

#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	­
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  
flatten/ReshapeReshapeblock5_pool/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:d*
dtype0
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:d*
dtype0
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:d*
dtype0
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:d*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:°
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:dx
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:d©
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:d
#batch_normalization/batchnorm/mul_1Muldense/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd§
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:d©
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d®
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_1/MatMulMatMulactivation/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block2_conv2_layer_call_and_return_conditional_losses_14297

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿH: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs

ª
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_15291
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_15291___redundant_placeholder0¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_15291___redundant_placeholder1¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_15291___redundant_placeholder2¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_15291___redundant_placeholder3S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
å
Kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: Í
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityOloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:


G__inference_block4_conv3_layer_call_and_return_conditional_losses_14418

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
^
B__inference_flatten_layer_call_and_return_conditional_losses_14517

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
G
+__inference_random_zoom_layer_call_fn_17659

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_14226j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block4_conv1_layer_call_and_return_conditional_losses_18695

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ú
¤
,__inference_block4_conv2_layer_call_fn_18704

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_14401x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼C
Ç
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_16941
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter¥
 random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsV
Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0¯
ªrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	S
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityU
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice­
¨random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Jrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderUrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPackRrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholdererandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
: 
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1PackNrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0grandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:­
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      £
Trandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceªrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0crandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0erandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0erandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maské
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcast]random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Urandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Á
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0^random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Æ
krandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderZrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ì
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Î
Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentityPrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ¡
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identity random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ð
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2IdentityPrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: û
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identity{random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "«
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityXrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"²
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slicerandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"Ø
¨random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceªrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±
G
+__inference_block5_pool_layer_call_fn_18850

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_14130
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ô
9loop_body_stateful_uniform_full_int_pfor_while_cond_18058n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1n
jloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_18058___redundant_placeholder0
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_18058___redundant_placeholder1
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_18058___redundant_placeholder2
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_18058___redundant_placeholder3;
7loop_body_stateful_uniform_full_int_pfor_while_identity

3loop_body/stateful_uniform_full_int/pfor/while/LessLess:loop_body_stateful_uniform_full_int_pfor_while_placeholderjloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity7loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:


G__inference_block3_conv1_layer_call_and_return_conditional_losses_18605

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
¥
Î
3__inference_batch_normalization_layer_call_fn_18898

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14157o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ú
¤
,__inference_block3_conv2_layer_call_fn_18614

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_14332x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs


G__inference_block2_conv1_layer_call_and_return_conditional_losses_14280

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿH@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@
 
_user_specified_nameinputs
9


Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_14999~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1{
wloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityE
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3y
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	~
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ï
:loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderEloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPackBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderUloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1Pack>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0Wloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0Sloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÉ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcastMloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Eloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
[loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderJloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ó
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :«
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ®
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ñ
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: °
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2Identity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: Û
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identitykloop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"ð
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slicewloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã

%__inference_dense_layer_call_fn_18875

inputs
unknown:
d
	unknown_0:d
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14529o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
¤
,__inference_block4_conv4_layer_call_fn_18744

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_14435x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±
G
+__inference_block3_pool_layer_call_fn_18670

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_14106
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ïw
è
__inference__traced_save_19217
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop)
%savev2_statevar_1_read_readvariableop	'
#savev2_statevar_read_readvariableop	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*±
value§B¤@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHð
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*
valueB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ð
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop%savev2_statevar_1_read_readvariableop#savev2_statevar_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@			
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ö
_input_shapesä
á: :@:@:@@:@:@::::::::::::::::::::::::::::
d:d:d:d:d:d:d:: : : : : ::: : : : :
d:d:d:d:d::
d:d:d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::&!"
 
_output_shapes
:
d: "

_output_shapes
:d: #

_output_shapes
:d: $

_output_shapes
:d: %

_output_shapes
:d: &

_output_shapes
:d:$' 

_output_shapes

:d: (

_output_shapes
::)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: : .

_output_shapes
:: /

_output_shapes
::0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :&4"
 
_output_shapes
:
d: 5

_output_shapes
:d: 6

_output_shapes
:d: 7

_output_shapes
:d:$8 

_output_shapes

:d: 9

_output_shapes
::&:"
 
_output_shapes
:
d: ;

_output_shapes
:d: <

_output_shapes
:d: =

_output_shapes
:d:$> 

_output_shapes

:d: ?

_output_shapes
::@

_output_shapes
: 
ú
¤
,__inference_block5_conv2_layer_call_fn_18794

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_14470x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
¢
F
*__inference_activation_layer_call_fn_18970

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_14549`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
åó
Õ'
 __inference__wrapped_model_14073
random_zoom_inputP
6sequential_block1_conv1_conv2d_readvariableop_resource:@E
7sequential_block1_conv1_biasadd_readvariableop_resource:@P
6sequential_block1_conv2_conv2d_readvariableop_resource:@@E
7sequential_block1_conv2_biasadd_readvariableop_resource:@Q
6sequential_block2_conv1_conv2d_readvariableop_resource:@F
7sequential_block2_conv1_biasadd_readvariableop_resource:	R
6sequential_block2_conv2_conv2d_readvariableop_resource:F
7sequential_block2_conv2_biasadd_readvariableop_resource:	R
6sequential_block3_conv1_conv2d_readvariableop_resource:F
7sequential_block3_conv1_biasadd_readvariableop_resource:	R
6sequential_block3_conv2_conv2d_readvariableop_resource:F
7sequential_block3_conv2_biasadd_readvariableop_resource:	R
6sequential_block3_conv3_conv2d_readvariableop_resource:F
7sequential_block3_conv3_biasadd_readvariableop_resource:	R
6sequential_block3_conv4_conv2d_readvariableop_resource:F
7sequential_block3_conv4_biasadd_readvariableop_resource:	R
6sequential_block4_conv1_conv2d_readvariableop_resource:F
7sequential_block4_conv1_biasadd_readvariableop_resource:	R
6sequential_block4_conv2_conv2d_readvariableop_resource:F
7sequential_block4_conv2_biasadd_readvariableop_resource:	R
6sequential_block4_conv3_conv2d_readvariableop_resource:F
7sequential_block4_conv3_biasadd_readvariableop_resource:	R
6sequential_block4_conv4_conv2d_readvariableop_resource:F
7sequential_block4_conv4_biasadd_readvariableop_resource:	R
6sequential_block5_conv1_conv2d_readvariableop_resource:F
7sequential_block5_conv1_biasadd_readvariableop_resource:	R
6sequential_block5_conv2_conv2d_readvariableop_resource:F
7sequential_block5_conv2_biasadd_readvariableop_resource:	R
6sequential_block5_conv3_conv2d_readvariableop_resource:F
7sequential_block5_conv3_biasadd_readvariableop_resource:	R
6sequential_block5_conv4_conv2d_readvariableop_resource:F
7sequential_block5_conv4_biasadd_readvariableop_resource:	C
/sequential_dense_matmul_readvariableop_resource:
d>
0sequential_dense_biasadd_readvariableop_resource:dI
;sequential_batch_normalization_cast_readvariableop_resource:dK
=sequential_batch_normalization_cast_1_readvariableop_resource:dK
=sequential_batch_normalization_cast_2_readvariableop_resource:dK
=sequential_batch_normalization_cast_3_readvariableop_resource:dC
1sequential_dense_1_matmul_readvariableop_resource:d@
2sequential_dense_1_biasadd_readvariableop_resource:
identity¢2sequential/batch_normalization/Cast/ReadVariableOp¢4sequential/batch_normalization/Cast_1/ReadVariableOp¢4sequential/batch_normalization/Cast_2/ReadVariableOp¢4sequential/batch_normalization/Cast_3/ReadVariableOp¢.sequential/block1_conv1/BiasAdd/ReadVariableOp¢-sequential/block1_conv1/Conv2D/ReadVariableOp¢.sequential/block1_conv2/BiasAdd/ReadVariableOp¢-sequential/block1_conv2/Conv2D/ReadVariableOp¢.sequential/block2_conv1/BiasAdd/ReadVariableOp¢-sequential/block2_conv1/Conv2D/ReadVariableOp¢.sequential/block2_conv2/BiasAdd/ReadVariableOp¢-sequential/block2_conv2/Conv2D/ReadVariableOp¢.sequential/block3_conv1/BiasAdd/ReadVariableOp¢-sequential/block3_conv1/Conv2D/ReadVariableOp¢.sequential/block3_conv2/BiasAdd/ReadVariableOp¢-sequential/block3_conv2/Conv2D/ReadVariableOp¢.sequential/block3_conv3/BiasAdd/ReadVariableOp¢-sequential/block3_conv3/Conv2D/ReadVariableOp¢.sequential/block3_conv4/BiasAdd/ReadVariableOp¢-sequential/block3_conv4/Conv2D/ReadVariableOp¢.sequential/block4_conv1/BiasAdd/ReadVariableOp¢-sequential/block4_conv1/Conv2D/ReadVariableOp¢.sequential/block4_conv2/BiasAdd/ReadVariableOp¢-sequential/block4_conv2/Conv2D/ReadVariableOp¢.sequential/block4_conv3/BiasAdd/ReadVariableOp¢-sequential/block4_conv3/Conv2D/ReadVariableOp¢.sequential/block4_conv4/BiasAdd/ReadVariableOp¢-sequential/block4_conv4/Conv2D/ReadVariableOp¢.sequential/block5_conv1/BiasAdd/ReadVariableOp¢-sequential/block5_conv1/Conv2D/ReadVariableOp¢.sequential/block5_conv2/BiasAdd/ReadVariableOp¢-sequential/block5_conv2/Conv2D/ReadVariableOp¢.sequential/block5_conv3/BiasAdd/ReadVariableOp¢-sequential/block5_conv3/Conv2D/ReadVariableOp¢.sequential/block5_conv4/BiasAdd/ReadVariableOp¢-sequential/block5_conv4/Conv2D/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¬
-sequential/block1_conv1/Conv2D/ReadVariableOpReadVariableOp6sequential_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ö
sequential/block1_conv1/Conv2DConv2Drandom_zoom_input5sequential/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
¢
.sequential/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp7sequential_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ç
sequential/block1_conv1/BiasAddBiasAdd'sequential/block1_conv1/Conv2D:output:06sequential/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential/block1_conv1/ReluRelu(sequential/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
-sequential/block1_conv2/Conv2D/ReadVariableOpReadVariableOp6sequential_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ï
sequential/block1_conv2/Conv2DConv2D*sequential/block1_conv1/Relu:activations:05sequential/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
¢
.sequential/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp7sequential_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ç
sequential/block1_conv2/BiasAddBiasAdd'sequential/block1_conv2/Conv2D:output:06sequential/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential/block1_conv2/ReluRelu(sequential/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ã
sequential/block1_pool/MaxPoolMaxPool*sequential/block1_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@*
ksize
*
paddingVALID*
strides
­
-sequential/block2_conv1/Conv2D/ReadVariableOpReadVariableOp6sequential_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ì
sequential/block2_conv1/Conv2DConv2D'sequential/block1_pool/MaxPool:output:05sequential/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
£
.sequential/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp7sequential_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
sequential/block2_conv1/BiasAddBiasAdd'sequential/block2_conv1/Conv2D:output:06sequential/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
sequential/block2_conv1/ReluRelu(sequential/block2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH®
-sequential/block2_conv2/Conv2D/ReadVariableOpReadVariableOp6sequential_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ï
sequential/block2_conv2/Conv2DConv2D*sequential/block2_conv1/Relu:activations:05sequential/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
£
.sequential/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp7sequential_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
sequential/block2_conv2/BiasAddBiasAdd'sequential/block2_conv2/Conv2D:output:06sequential/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
sequential/block2_conv2/ReluRelu(sequential/block2_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿHÃ
sequential/block2_pool/MaxPoolMaxPool*sequential/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
ksize
*
paddingVALID*
strides
®
-sequential/block3_conv1/Conv2D/ReadVariableOpReadVariableOp6sequential_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
sequential/block3_conv1/Conv2DConv2D'sequential/block2_pool/MaxPool:output:05sequential/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
£
.sequential/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp7sequential_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block3_conv1/BiasAddBiasAdd'sequential/block3_conv1/Conv2D:output:06sequential/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
sequential/block3_conv1/ReluRelu(sequential/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@®
-sequential/block3_conv2/Conv2D/ReadVariableOpReadVariableOp6sequential_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block3_conv2/Conv2DConv2D*sequential/block3_conv1/Relu:activations:05sequential/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
£
.sequential/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp7sequential_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block3_conv2/BiasAddBiasAdd'sequential/block3_conv2/Conv2D:output:06sequential/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
sequential/block3_conv2/ReluRelu(sequential/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@®
-sequential/block3_conv3/Conv2D/ReadVariableOpReadVariableOp6sequential_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block3_conv3/Conv2DConv2D*sequential/block3_conv2/Relu:activations:05sequential/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
£
.sequential/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp7sequential_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block3_conv3/BiasAddBiasAdd'sequential/block3_conv3/Conv2D:output:06sequential/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
sequential/block3_conv3/ReluRelu(sequential/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@®
-sequential/block3_conv4/Conv2D/ReadVariableOpReadVariableOp6sequential_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block3_conv4/Conv2DConv2D*sequential/block3_conv3/Relu:activations:05sequential/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
£
.sequential/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp7sequential_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block3_conv4/BiasAddBiasAdd'sequential/block3_conv4/Conv2D:output:06sequential/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
sequential/block3_conv4/ReluRelu(sequential/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Ã
sequential/block3_pool/MaxPoolMaxPool*sequential/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
®
-sequential/block4_conv1/Conv2D/ReadVariableOpReadVariableOp6sequential_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
sequential/block4_conv1/Conv2DConv2D'sequential/block3_pool/MaxPool:output:05sequential/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
£
.sequential/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp7sequential_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block4_conv1/BiasAddBiasAdd'sequential/block4_conv1/Conv2D:output:06sequential/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential/block4_conv1/ReluRelu(sequential/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
-sequential/block4_conv2/Conv2D/ReadVariableOpReadVariableOp6sequential_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block4_conv2/Conv2DConv2D*sequential/block4_conv1/Relu:activations:05sequential/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
£
.sequential/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp7sequential_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block4_conv2/BiasAddBiasAdd'sequential/block4_conv2/Conv2D:output:06sequential/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential/block4_conv2/ReluRelu(sequential/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
-sequential/block4_conv3/Conv2D/ReadVariableOpReadVariableOp6sequential_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block4_conv3/Conv2DConv2D*sequential/block4_conv2/Relu:activations:05sequential/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
£
.sequential/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp7sequential_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block4_conv3/BiasAddBiasAdd'sequential/block4_conv3/Conv2D:output:06sequential/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential/block4_conv3/ReluRelu(sequential/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ®
-sequential/block4_conv4/Conv2D/ReadVariableOpReadVariableOp6sequential_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block4_conv4/Conv2DConv2D*sequential/block4_conv3/Relu:activations:05sequential/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
£
.sequential/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp7sequential_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block4_conv4/BiasAddBiasAdd'sequential/block4_conv4/Conv2D:output:06sequential/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sequential/block4_conv4/ReluRelu(sequential/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ã
sequential/block4_pool/MaxPoolMaxPool*sequential/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
ksize
*
paddingVALID*
strides
®
-sequential/block5_conv1/Conv2D/ReadVariableOpReadVariableOp6sequential_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ë
sequential/block5_conv1/Conv2DConv2D'sequential/block4_pool/MaxPool:output:05sequential/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
£
.sequential/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp7sequential_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block5_conv1/BiasAddBiasAdd'sequential/block5_conv1/Conv2D:output:06sequential/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
sequential/block5_conv1/ReluRelu(sequential/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	®
-sequential/block5_conv2/Conv2D/ReadVariableOpReadVariableOp6sequential_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block5_conv2/Conv2DConv2D*sequential/block5_conv1/Relu:activations:05sequential/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
£
.sequential/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp7sequential_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block5_conv2/BiasAddBiasAdd'sequential/block5_conv2/Conv2D:output:06sequential/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
sequential/block5_conv2/ReluRelu(sequential/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	®
-sequential/block5_conv3/Conv2D/ReadVariableOpReadVariableOp6sequential_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block5_conv3/Conv2DConv2D*sequential/block5_conv2/Relu:activations:05sequential/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
£
.sequential/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp7sequential_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block5_conv3/BiasAddBiasAdd'sequential/block5_conv3/Conv2D:output:06sequential/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
sequential/block5_conv3/ReluRelu(sequential/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	®
-sequential/block5_conv4/Conv2D/ReadVariableOpReadVariableOp6sequential_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0î
sequential/block5_conv4/Conv2DConv2D*sequential/block5_conv3/Relu:activations:05sequential/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
£
.sequential/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp7sequential_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Æ
sequential/block5_conv4/BiasAddBiasAdd'sequential/block5_conv4/Conv2D:output:06sequential/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
sequential/block5_conv4/ReluRelu(sequential/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Ã
sequential/block5_pool/MaxPoolMaxPool*sequential/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  ¥
sequential/flatten/ReshapeReshape'sequential/block5_pool/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0¨
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
2sequential/batch_normalization/Cast/ReadVariableOpReadVariableOp;sequential_batch_normalization_cast_readvariableop_resource*
_output_shapes
:d*
dtype0®
4sequential/batch_normalization/Cast_1/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:d*
dtype0®
4sequential/batch_normalization/Cast_2/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:d*
dtype0®
4sequential/batch_normalization/Cast_3/ReadVariableOpReadVariableOp=sequential_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:d*
dtype0s
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ñ
,sequential/batch_normalization/batchnorm/addAddV2<sequential/batch_normalization/Cast_1/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:d
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:dÊ
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0<sequential/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:d¼
.sequential/batch_normalization/batchnorm/mul_1Mul!sequential/dense/BiasAdd:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÈ
.sequential/batch_normalization/batchnorm/mul_2Mul:sequential/batch_normalization/Cast/ReadVariableOp:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:dÊ
,sequential/batch_normalization/batchnorm/subSub<sequential/batch_normalization/Cast_2/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:dÏ
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential/activation/ReluRelu2sequential/batch_normalization/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0±
sequential/dense_1/MatMulMatMul(sequential/activation/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
IdentityIdentity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
NoOpNoOp3^sequential/batch_normalization/Cast/ReadVariableOp5^sequential/batch_normalization/Cast_1/ReadVariableOp5^sequential/batch_normalization/Cast_2/ReadVariableOp5^sequential/batch_normalization/Cast_3/ReadVariableOp/^sequential/block1_conv1/BiasAdd/ReadVariableOp.^sequential/block1_conv1/Conv2D/ReadVariableOp/^sequential/block1_conv2/BiasAdd/ReadVariableOp.^sequential/block1_conv2/Conv2D/ReadVariableOp/^sequential/block2_conv1/BiasAdd/ReadVariableOp.^sequential/block2_conv1/Conv2D/ReadVariableOp/^sequential/block2_conv2/BiasAdd/ReadVariableOp.^sequential/block2_conv2/Conv2D/ReadVariableOp/^sequential/block3_conv1/BiasAdd/ReadVariableOp.^sequential/block3_conv1/Conv2D/ReadVariableOp/^sequential/block3_conv2/BiasAdd/ReadVariableOp.^sequential/block3_conv2/Conv2D/ReadVariableOp/^sequential/block3_conv3/BiasAdd/ReadVariableOp.^sequential/block3_conv3/Conv2D/ReadVariableOp/^sequential/block3_conv4/BiasAdd/ReadVariableOp.^sequential/block3_conv4/Conv2D/ReadVariableOp/^sequential/block4_conv1/BiasAdd/ReadVariableOp.^sequential/block4_conv1/Conv2D/ReadVariableOp/^sequential/block4_conv2/BiasAdd/ReadVariableOp.^sequential/block4_conv2/Conv2D/ReadVariableOp/^sequential/block4_conv3/BiasAdd/ReadVariableOp.^sequential/block4_conv3/Conv2D/ReadVariableOp/^sequential/block4_conv4/BiasAdd/ReadVariableOp.^sequential/block4_conv4/Conv2D/ReadVariableOp/^sequential/block5_conv1/BiasAdd/ReadVariableOp.^sequential/block5_conv1/Conv2D/ReadVariableOp/^sequential/block5_conv2/BiasAdd/ReadVariableOp.^sequential/block5_conv2/Conv2D/ReadVariableOp/^sequential/block5_conv3/BiasAdd/ReadVariableOp.^sequential/block5_conv3/Conv2D/ReadVariableOp/^sequential/block5_conv4/BiasAdd/ReadVariableOp.^sequential/block5_conv4/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2sequential/batch_normalization/Cast/ReadVariableOp2sequential/batch_normalization/Cast/ReadVariableOp2l
4sequential/batch_normalization/Cast_1/ReadVariableOp4sequential/batch_normalization/Cast_1/ReadVariableOp2l
4sequential/batch_normalization/Cast_2/ReadVariableOp4sequential/batch_normalization/Cast_2/ReadVariableOp2l
4sequential/batch_normalization/Cast_3/ReadVariableOp4sequential/batch_normalization/Cast_3/ReadVariableOp2`
.sequential/block1_conv1/BiasAdd/ReadVariableOp.sequential/block1_conv1/BiasAdd/ReadVariableOp2^
-sequential/block1_conv1/Conv2D/ReadVariableOp-sequential/block1_conv1/Conv2D/ReadVariableOp2`
.sequential/block1_conv2/BiasAdd/ReadVariableOp.sequential/block1_conv2/BiasAdd/ReadVariableOp2^
-sequential/block1_conv2/Conv2D/ReadVariableOp-sequential/block1_conv2/Conv2D/ReadVariableOp2`
.sequential/block2_conv1/BiasAdd/ReadVariableOp.sequential/block2_conv1/BiasAdd/ReadVariableOp2^
-sequential/block2_conv1/Conv2D/ReadVariableOp-sequential/block2_conv1/Conv2D/ReadVariableOp2`
.sequential/block2_conv2/BiasAdd/ReadVariableOp.sequential/block2_conv2/BiasAdd/ReadVariableOp2^
-sequential/block2_conv2/Conv2D/ReadVariableOp-sequential/block2_conv2/Conv2D/ReadVariableOp2`
.sequential/block3_conv1/BiasAdd/ReadVariableOp.sequential/block3_conv1/BiasAdd/ReadVariableOp2^
-sequential/block3_conv1/Conv2D/ReadVariableOp-sequential/block3_conv1/Conv2D/ReadVariableOp2`
.sequential/block3_conv2/BiasAdd/ReadVariableOp.sequential/block3_conv2/BiasAdd/ReadVariableOp2^
-sequential/block3_conv2/Conv2D/ReadVariableOp-sequential/block3_conv2/Conv2D/ReadVariableOp2`
.sequential/block3_conv3/BiasAdd/ReadVariableOp.sequential/block3_conv3/BiasAdd/ReadVariableOp2^
-sequential/block3_conv3/Conv2D/ReadVariableOp-sequential/block3_conv3/Conv2D/ReadVariableOp2`
.sequential/block3_conv4/BiasAdd/ReadVariableOp.sequential/block3_conv4/BiasAdd/ReadVariableOp2^
-sequential/block3_conv4/Conv2D/ReadVariableOp-sequential/block3_conv4/Conv2D/ReadVariableOp2`
.sequential/block4_conv1/BiasAdd/ReadVariableOp.sequential/block4_conv1/BiasAdd/ReadVariableOp2^
-sequential/block4_conv1/Conv2D/ReadVariableOp-sequential/block4_conv1/Conv2D/ReadVariableOp2`
.sequential/block4_conv2/BiasAdd/ReadVariableOp.sequential/block4_conv2/BiasAdd/ReadVariableOp2^
-sequential/block4_conv2/Conv2D/ReadVariableOp-sequential/block4_conv2/Conv2D/ReadVariableOp2`
.sequential/block4_conv3/BiasAdd/ReadVariableOp.sequential/block4_conv3/BiasAdd/ReadVariableOp2^
-sequential/block4_conv3/Conv2D/ReadVariableOp-sequential/block4_conv3/Conv2D/ReadVariableOp2`
.sequential/block4_conv4/BiasAdd/ReadVariableOp.sequential/block4_conv4/BiasAdd/ReadVariableOp2^
-sequential/block4_conv4/Conv2D/ReadVariableOp-sequential/block4_conv4/Conv2D/ReadVariableOp2`
.sequential/block5_conv1/BiasAdd/ReadVariableOp.sequential/block5_conv1/BiasAdd/ReadVariableOp2^
-sequential/block5_conv1/Conv2D/ReadVariableOp-sequential/block5_conv1/Conv2D/ReadVariableOp2`
.sequential/block5_conv2/BiasAdd/ReadVariableOp.sequential/block5_conv2/BiasAdd/ReadVariableOp2^
-sequential/block5_conv2/Conv2D/ReadVariableOp-sequential/block5_conv2/Conv2D/ReadVariableOp2`
.sequential/block5_conv3/BiasAdd/ReadVariableOp.sequential/block5_conv3/BiasAdd/ReadVariableOp2^
-sequential/block5_conv3/Conv2D/ReadVariableOp-sequential/block5_conv3/Conv2D/ReadVariableOp2`
.sequential/block5_conv4/BiasAdd/ReadVariableOp.sequential/block5_conv4/BiasAdd/ReadVariableOp2^
-sequential/block5_conv4/Conv2D/ReadVariableOp-sequential/block5_conv4/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_zoom_input
ØQ
ô
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_15223§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2¤
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityY
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4¢
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice	
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :«
Nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderYloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¢
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ×
^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¤
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ×
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1PackRloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0kloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:±
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounteraloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDimsmloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Ö
oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ç
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimsqloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0dloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:Ú
qloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ü
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Ö
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: ­
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1Identity¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ø
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2IdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"Â
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedsliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ñ

E__inference_sequential_layer_call_and_return_conditional_losses_14569

inputs,
block1_conv1_14246:@ 
block1_conv1_14248:@,
block1_conv2_14263:@@ 
block1_conv2_14265:@-
block2_conv1_14281:@!
block2_conv1_14283:	.
block2_conv2_14298:!
block2_conv2_14300:	.
block3_conv1_14316:!
block3_conv1_14318:	.
block3_conv2_14333:!
block3_conv2_14335:	.
block3_conv3_14350:!
block3_conv3_14352:	.
block3_conv4_14367:!
block3_conv4_14369:	.
block4_conv1_14385:!
block4_conv1_14387:	.
block4_conv2_14402:!
block4_conv2_14404:	.
block4_conv3_14419:!
block4_conv3_14421:	.
block4_conv4_14436:!
block4_conv4_14438:	.
block5_conv1_14454:!
block5_conv1_14456:	.
block5_conv2_14471:!
block5_conv2_14473:	.
block5_conv3_14488:!
block5_conv3_14490:	.
block5_conv4_14505:!
block5_conv4_14507:	
dense_14530:
d
dense_14532:d'
batch_normalization_14535:d'
batch_normalization_14537:d'
batch_normalization_14539:d'
batch_normalization_14541:d
dense_1_14563:d
dense_1_14565:
identity¢+batch_normalization/StatefulPartitionedCall¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÊ
random_zoom/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_14226ð
random_contrast/PartitionedCallPartitionedCall$random_zoom/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_14232¬
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall(random_contrast/PartitionedCall:output:0block1_conv1_14246block1_conv1_14248*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_14245±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_14263block1_conv2_14265*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_14262ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_14082¨
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_14281block2_conv1_14283*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_14280±
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_14298block2_conv2_14300*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_14297ð
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_14094§
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_14316block3_conv1_14318*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_14315°
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_14333block3_conv2_14335*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_14332°
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_14350block3_conv3_14352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_14349°
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_14367block3_conv4_14369*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_14366ð
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_14106§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_14385block4_conv1_14387*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_14384°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_14402block4_conv2_14404*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_14401°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_14419block4_conv3_14421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_14418°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_14436block4_conv4_14438*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_14435ð
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_14118§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_14454block5_conv1_14456*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_14453°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_14471block5_conv2_14473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_14470°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_14488block5_conv3_14490*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_14487°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_14505block5_conv4_14507*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_14504ð
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_14130Ø
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14517þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_14530dense_14532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14529ö
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_14535batch_normalization_14537batch_normalization_14539batch_normalization_14541*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14157ì
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_14549
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_14563dense_1_14565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14562w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

*__inference_sequential_layer_call_fn_16048
random_zoom_input
unknown:	
	unknown_0:	#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	&

unknown_31:

unknown_32:	

unknown_33:
d

unknown_34:d

unknown_35:d

unknown_36:d

unknown_37:d

unknown_38:d

unknown_39:d

unknown_40:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_zoom_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$'()**0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_15872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_zoom_input


G__inference_block4_conv3_layer_call_and_return_conditional_losses_18735

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


G__inference_block1_conv2_layer_call_and_return_conditional_losses_18525

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_block5_conv2_layer_call_and_return_conditional_losses_14470

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
ú
¤
,__inference_block3_conv1_layer_call_fn_18594

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_14315x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
È
^
B__inference_flatten_layer_call_and_return_conditional_losses_18866

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_random_contrast_layer_call_and_return_conditional_losses_17788

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
¤
,__inference_block2_conv2_layer_call_fn_18564

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_14297y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿH: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
¯$
Ï
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18965

inputs5
'assignmovingavg_readvariableop_resource:d7
)assignmovingavg_1_readvariableop_resource:d*
cast_readvariableop_resource:d,
cast_1_readvariableop_resource:d
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:d*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:dx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:d¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:d*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:d~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:d´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:d*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:dm
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:dk
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÞ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
­
á	
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_14933
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_14933___redundant_placeholder0¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_14933___redundant_placeholder1¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_14933___redundant_placeholder2J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity
Á
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessIloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: »
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityFloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:


Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_17007£
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter©
¤random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderZ
Vrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1£
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_sliceº
µrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_17007___redundant_placeholder0	U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity
í
Mrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLessTrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: Ñ
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityQrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityZrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ØQ
ô
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_18159§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2¤
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityY
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4¢
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice	
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :«
Nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderYloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¢
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ×
^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¤
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ×
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1PackRloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0kloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:±
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounteraloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDimsmloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Ö
oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ç
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimsqloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0dloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:Ú
qloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ü
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Ö
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: ­
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1Identity¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ø
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2IdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"Â
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedsliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷:

Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_17870
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0
{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityL
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	l
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xl
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1¢Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipÏ
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkip{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¬
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsTloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0Uloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:¢
bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemKloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderQloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :È
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterNloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityGloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ë
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsC^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2IdentityEloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ®
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identityrloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ó
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOpM^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xjloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0"
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"ø
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
û
¡
,__inference_block1_conv1_layer_call_fn_18494

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_14245y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
	
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_15222§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¾
¹loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_15222___redundant_placeholder0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
õ
Oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: Õ
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitySloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
²

Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_17064
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counter
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterationsN
Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderP
Lrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_17064___redundant_placeholder0¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_17064___redundant_placeholder1¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_17064___redundant_placeholder2¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_17064___redundant_placeholder3K
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity
Å
Crandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/LessLessJrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: ½
Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentityGrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityPrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
ú
¤
,__inference_block5_conv1_layer_call_fn_18774

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_14453x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
ØP

9loop_body_stateful_uniform_full_int_pfor_while_body_15123n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1k
gloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0^
Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0;
7loop_body_stateful_uniform_full_int_pfor_while_identity=
9loop_body_stateful_uniform_full_int_pfor_while_identity_1=
9loop_body_stateful_uniform_full_int_pfor_while_identity_2=
9loop_body_stateful_uniform_full_int_pfor_while_identity_3i
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZ
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algv
4loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :×
2loop_body/stateful_uniform_full_int/pfor/while/addAddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder=loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Bloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderMloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1Pack6loop_body/stateful_uniform_full_int/pfor/while/add:z:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
<loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0Kloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskx
6loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderOloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1Pack8loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0Qloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¥
>loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maské
Nloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0Eloop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0Gloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
=loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
9loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsWloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0Floop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
Sloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1:loop_body_stateful_uniform_full_int_pfor_while_placeholderBloop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐx
6loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: x
6loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
4loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_counter?loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity8loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: Ø
9loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identityploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
:  
9loop_body/stateful_uniform_full_int/pfor/while/Identity_2Identity8loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: Ë
9loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identitycloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_1Bloop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_2Bloop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_3Bloop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0"²
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algXloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0"Ð
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slicegloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"¶
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
ú
¤
,__inference_block5_conv3_layer_call_fn_18814

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_14487x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
¢
Ê
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_17233¿
ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterÅ
Àrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsf
brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderh
drandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1¿
ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceÖ
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_17233___redundant_placeholder0Ö
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_17233___redundant_placeholder1Ö
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_17233___redundant_placeholder2Ö
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_17233___redundant_placeholder3c
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
¥
[random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessbrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: í
_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentity_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Ë
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityhrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:


G__inference_block5_conv3_layer_call_and_return_conditional_losses_18825

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs


G__inference_block3_conv4_layer_call_and_return_conditional_losses_14366

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
÷:

Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_14934
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0
{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityL
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	l
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xl
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1¢Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipÏ
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkip{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¬
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsTloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0Uloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:¢
bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemKloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderQloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :È
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterNloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityGloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ë
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsC^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2IdentityEloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ®
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identityrloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ó
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOpM^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xjloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0"
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"ø
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


G__inference_block4_conv2_layer_call_and_return_conditional_losses_14401

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

f
J__inference_random_contrast_layer_call_and_return_conditional_losses_14232

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
ÿ
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_16940
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter¥
 random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsV
Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice¶
±random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_16940___redundant_placeholder0	S
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity
å
Krandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLessRrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: Í
Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentityOrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "«
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityXrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
Ìö
·'
!__inference__traced_restore_19416
file_prefix>
$assignvariableop_block1_conv1_kernel:@2
$assignvariableop_1_block1_conv1_bias:@@
&assignvariableop_2_block1_conv2_kernel:@@2
$assignvariableop_3_block1_conv2_bias:@A
&assignvariableop_4_block2_conv1_kernel:@3
$assignvariableop_5_block2_conv1_bias:	B
&assignvariableop_6_block2_conv2_kernel:3
$assignvariableop_7_block2_conv2_bias:	B
&assignvariableop_8_block3_conv1_kernel:3
$assignvariableop_9_block3_conv1_bias:	C
'assignvariableop_10_block3_conv2_kernel:4
%assignvariableop_11_block3_conv2_bias:	C
'assignvariableop_12_block3_conv3_kernel:4
%assignvariableop_13_block3_conv3_bias:	C
'assignvariableop_14_block3_conv4_kernel:4
%assignvariableop_15_block3_conv4_bias:	C
'assignvariableop_16_block4_conv1_kernel:4
%assignvariableop_17_block4_conv1_bias:	C
'assignvariableop_18_block4_conv2_kernel:4
%assignvariableop_19_block4_conv2_bias:	C
'assignvariableop_20_block4_conv3_kernel:4
%assignvariableop_21_block4_conv3_bias:	C
'assignvariableop_22_block4_conv4_kernel:4
%assignvariableop_23_block4_conv4_bias:	C
'assignvariableop_24_block5_conv1_kernel:4
%assignvariableop_25_block5_conv1_bias:	C
'assignvariableop_26_block5_conv2_kernel:4
%assignvariableop_27_block5_conv2_bias:	C
'assignvariableop_28_block5_conv3_kernel:4
%assignvariableop_29_block5_conv3_bias:	C
'assignvariableop_30_block5_conv4_kernel:4
%assignvariableop_31_block5_conv4_bias:	4
 assignvariableop_32_dense_kernel:
d,
assignvariableop_33_dense_bias:d;
-assignvariableop_34_batch_normalization_gamma:d:
,assignvariableop_35_batch_normalization_beta:dA
3assignvariableop_36_batch_normalization_moving_mean:dE
7assignvariableop_37_batch_normalization_moving_variance:d4
"assignvariableop_38_dense_1_kernel:d.
 assignvariableop_39_dense_1_bias:'
assignvariableop_40_adam_iter:	 )
assignvariableop_41_adam_beta_1: )
assignvariableop_42_adam_beta_2: (
assignvariableop_43_adam_decay: 0
&assignvariableop_44_adam_learning_rate: ,
assignvariableop_45_statevar_1:	*
assignvariableop_46_statevar:	%
assignvariableop_47_total_1: %
assignvariableop_48_count_1: #
assignvariableop_49_total: #
assignvariableop_50_count: ;
'assignvariableop_51_adam_dense_kernel_m:
d3
%assignvariableop_52_adam_dense_bias_m:dB
4assignvariableop_53_adam_batch_normalization_gamma_m:dA
3assignvariableop_54_adam_batch_normalization_beta_m:d;
)assignvariableop_55_adam_dense_1_kernel_m:d5
'assignvariableop_56_adam_dense_1_bias_m:;
'assignvariableop_57_adam_dense_kernel_v:
d3
%assignvariableop_58_adam_dense_bias_v:dB
4assignvariableop_59_adam_batch_normalization_gamma_v:dA
3assignvariableop_60_adam_batch_normalization_beta_v:d;
)assignvariableop_61_adam_dense_1_kernel_v:d5
'assignvariableop_62_adam_dense_1_bias_v:
identity_64¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*±
value§B¤@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHó
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*
valueB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block4_conv3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block4_conv3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block5_conv2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block5_conv2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block5_conv3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block5_conv3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block5_conv4_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp%assignvariableop_31_block5_conv4_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_dense_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp-assignvariableop_34_batch_normalization_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp,assignvariableop_35_batch_normalization_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_36AssignVariableOp3assignvariableop_36_batch_normalization_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp assignvariableop_39_dense_1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_iterIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_decayIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_learning_rateIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_45AssignVariableOpassignvariableop_45_statevar_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_46AssignVariableOpassignvariableop_46_statevarIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_dense_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_batch_normalization_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_batch_normalization_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_dense_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adam_dense_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_batch_normalization_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_batch_normalization_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¹
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: ¦
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_64Identity_64:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

©
J__inference_random_contrast_layer_call_and_return_conditional_losses_15549

inputsI
;loop_body_stateful_uniform_full_int_rngreadandskip_resource:	
identity¢2loop_body/stateful_uniform_full_int/RngReadAndSkip¢=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:s
)loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(loop_body/stateful_uniform_full_int/ProdProd2loop_body/stateful_uniform_full_int/shape:output:02loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*loop_body/stateful_uniform_full_int/Cast_1Cast1loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
7loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/stateful_uniform_full_int/strided_sliceStridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+loop_body/stateful_uniform_full_int/BitcastBitcast:loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3loop_body/stateful_uniform_full_int/strided_slice_1StridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-loop_body/stateful_uniform_full_int/Bitcast_1Bitcast<loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Ã
#loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV22loop_body/stateful_uniform_full_int/shape:output:06loop_body/stateful_uniform_full_int/Bitcast_1:output:04loop_body/stateful_uniform_full_int/Bitcast:output:00loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 
loop_body/stackPack,loop_body/stateful_uniform_full_int:output:0loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
loop_body/strided_slice_1StridedSliceloop_body/stack:output:0(loop_body/strided_slice_1/stack:output:0*loop_body/strided_slice_1/stack_1:output:0*loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskk
(loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?k
&loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *33³?¥
?loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"loop_body/strided_slice_1:output:0* 
_output_shapes
::
?loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21loop_body/stateless_random_uniform/shape:output:0Eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&loop_body/stateless_random_uniform/subSub/loop_body/stateless_random_uniform/max:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&loop_body/stateless_random_uniform/mulMulDloop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"loop_body/stateless_random_uniformAddV2*loop_body/stateless_random_uniform/mul:z:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
loop_body/adjust_contrastAdjustContrastv2loop_body/GatherV2:output:0&loop_body/stateless_random_uniform:z:0*$
_output_shapes
:
"loop_body/adjust_contrast/IdentityIdentity"loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:f
!loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C²
loop_body/clip_by_value/MinimumMinimum+loop_body/adjust_contrast/Identity:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:^
loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
loop_body/clip_by_valueMaximum#loop_body/clip_by_value/Minimum:z:0"loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:\
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Kloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Tloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Sloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReserve\loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Ploop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhileSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0Yloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:03^loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *T
bodyLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_14934*T
condLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_14933*#
output_shapes
: : : : : : : : 
?loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ©
Xloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ´
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Bloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
=loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2Oloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Kloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:å
Cloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Mloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReserveUloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌx
6loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Iloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
6loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhileLloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0Rloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0?loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0Lloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *M
bodyERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_14999*M
condERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_14998*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ{
8loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¢
Qloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2?loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3Zloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Aloop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2Qloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Mloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
Eloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Oloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿË
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReserveWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌz
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Kloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ®
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhileNloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0Tloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *O
bodyGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_15066*O
condGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_15065*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ}
:loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¤
Sloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3\loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
<loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ü
6loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Eloop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ­
6loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReserveMloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐp
.loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ}
;loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë
.loop_body/stateful_uniform_full_int/pfor/whileStatelessWhileDloop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0Jloop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:07loop_body/stateful_uniform_full_int/pfor/Const:output:0?loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:02loop_body/stateful_uniform_full_int/shape:output:00loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *E
body=R;
9loop_body_stateful_uniform_full_int_pfor_while_body_15123*E
cond=R;
9loop_body_stateful_uniform_full_int_pfor_while_cond_15122*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: s
0loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Iloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿø
;loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV27loop_body/stateful_uniform_full_int/pfor/while:output:3Rloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:09loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0d
loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:f
$loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :¹
loop_body/stack/pfor/ones_likeFill=loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0-loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:u
"loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¢
loop_body/stack/pfor/ReshapeReshape'loop_body/stack/pfor/ones_like:output:0+loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
$loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
loop_body/stack/pfor/Reshape_1Reshapepfor/Reshape:output:0-loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:b
 loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
loop_body/stack/pfor/concatConcatV2'loop_body/stack/pfor/Reshape_1:output:0%loop_body/stack/pfor/Reshape:output:0)loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:e
#loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : £
loop_body/stack/pfor/ExpandDims
ExpandDimsloop_body/zeros_like:output:0,loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:£
loop_body/stack/pfor/TileTile(loop_body/stack/pfor/ExpandDims:output:0$loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
loop_body/stack/pfor/stackPackDloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:0"loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axisx
.loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: l
*loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
%loop_body/strided_slice_1/pfor/concatConcatV27loop_body/strided_slice_1/pfor/concat/values_0:output:0(loop_body/strided_slice_1/stack:output:03loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: n
,loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_1ConcatV29loop_body/strided_slice_1/pfor/concat_1/values_0:output:0*loop_body/strided_slice_1/stack_1:output:05loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:n
,loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_2ConcatV29loop_body/strided_slice_1/pfor/concat_2/values_0:output:0*loop_body/strided_slice_1/stack_2:output:05loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
+loop_body/strided_slice_1/pfor/StridedSliceStridedSlice#loop_body/stack/pfor/stack:output:0.loop_body/strided_slice_1/pfor/concat:output:00loop_body/strided_slice_1/pfor/concat_1:output:00loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask¢
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlicepfor/Reshape:output:0aloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask«
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReserveiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ­
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Tloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReservekloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¨
]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhile`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0floop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:04loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *a
bodyYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_15223*a
condYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_15222*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¶
eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   è
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ¸
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlicepfor/Reshape:output:0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask§
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserveeloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¤
Yloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Æ

Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhile\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0bloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:01loop_body/stateless_random_uniform/shape:output:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *]
bodyURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_15292*]
condURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_15291*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 
Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ´
aloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÔ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0r
0loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :t
2loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : s
1loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ò
/loop_body/stateless_random_uniform/mul/pfor/addAddV2;loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0:loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: Ï
3loop_body/stateless_random_uniform/mul/pfor/MaximumMaximum3loop_body/stateless_random_uniform/mul/pfor/add:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ½
1loop_body/stateless_random_uniform/mul/pfor/ShapeShape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:Ë
/loop_body/stateless_random_uniform/mul/pfor/subSub7loop_body/stateless_random_uniform/mul/pfor/Maximum:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
9loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ü
3loop_body/stateless_random_uniform/mul/pfor/ReshapeReshape3loop_body/stateless_random_uniform/mul/pfor/sub:z:0Bloop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
6loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ú
0loop_body/stateless_random_uniform/mul/pfor/TileTile?loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0<loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
?loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Hloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
;loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masky
7loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
2loop_body/stateless_random_uniform/mul/pfor/concatConcatV2Bloop_body/stateless_random_uniform/mul/pfor/strided_slice:output:09loop_body/stateless_random_uniform/mul/pfor/Tile:output:0Dloop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0@loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
5loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0;loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
/loop_body/stateless_random_uniform/mul/pfor/MulMul>loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :p
.loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : o
-loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Æ
+loop_body/stateless_random_uniform/pfor/addAddV27loop_body/stateless_random_uniform/pfor/Rank_1:output:06loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: Ã
/loop_body/stateless_random_uniform/pfor/MaximumMaximum/loop_body/stateless_random_uniform/pfor/add:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
-loop_body/stateless_random_uniform/pfor/ShapeShape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:¿
+loop_body/stateless_random_uniform/pfor/subSub3loop_body/stateless_random_uniform/pfor/Maximum:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
5loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/loop_body/stateless_random_uniform/pfor/ReshapeReshape/loop_body/stateless_random_uniform/pfor/sub:z:0>loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,loop_body/stateless_random_uniform/pfor/TileTile;loop_body/stateless_random_uniform/pfor/Tile/input:output:08loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5loop_body/stateless_random_uniform/pfor/strided_sliceStridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Dloop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Floop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.loop_body/stateless_random_uniform/pfor/concatConcatV2>loop_body/stateless_random_uniform/pfor/strided_slice:output:05loop_body/stateless_random_uniform/pfor/Tile:output:0@loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0<loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ø
1loop_body/stateless_random_uniform/pfor/Reshape_1Reshape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:07loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
-loop_body/stateless_random_uniform/pfor/AddV2AddV2:loop_body/stateless_random_uniform/pfor/Reshape_1:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : î
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
,loop_body/adjust_contrast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0;loop_body/adjust_contrast/pfor/strided_slice/stack:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserveCloop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:05loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
$loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
7loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ï
$loop_body/adjust_contrast/pfor/whileStatelessWhile:loop_body/adjust_contrast/pfor/while/loop_counter:output:0@loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0-loop_body/adjust_contrast/pfor/Const:output:05loop_body/adjust_contrast/pfor/TensorArrayV2:handle:05loop_body/adjust_contrast/pfor/strided_slice:output:0)loop_body/GatherV2/pfor/GatherV2:output:01loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *;
body3R1
/loop_body_adjust_contrast_pfor_while_body_15441*;
cond3R1
/loop_body_adjust_contrast_pfor_while_cond_15440*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿi
&loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
?loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         Ú
1loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2-loop_body/adjust_contrast/pfor/while:output:3Hloop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0´
0loop_body/adjust_contrast/Identity/pfor/IdentityIdentity:loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/clip_by_value/Minimum/pfor/addAddV24loop_body/clip_by_value/Minimum/pfor/Rank_1:output:03loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/clip_by_value/Minimum/pfor/MaximumMaximum,loop_body/clip_by_value/Minimum/pfor/add:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/clip_by_value/Minimum/pfor/ShapeShape9loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:¶
(loop_body/clip_by_value/Minimum/pfor/subSub0loop_body/clip_by_value/Minimum/pfor/Maximum:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/clip_by_value/Minimum/pfor/ReshapeReshape,loop_body/clip_by_value/Minimum/pfor/sub:z:0;loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/clip_by_value/Minimum/pfor/TileTile8loop_body/clip_by_value/Minimum/pfor/Tile/input:output:05loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Aloop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/clip_by_value/Minimum/pfor/concatConcatV2;loop_body/clip_by_value/Minimum/pfor/strided_slice:output:02loop_body/clip_by_value/Minimum/pfor/Tile:output:0=loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:09loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:æ
.loop_body/clip_by_value/Minimum/pfor/Reshape_1Reshape9loop_body/adjust_contrast/Identity/pfor/Identity:output:04loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
,loop_body/clip_by_value/Minimum/pfor/MinimumMinimum7loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : d
"loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¥
 loop_body/clip_by_value/pfor/addAddV2,loop_body/clip_by_value/pfor/Rank_1:output:0+loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: ¢
$loop_body/clip_by_value/pfor/MaximumMaximum$loop_body/clip_by_value/pfor/add:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
"loop_body/clip_by_value/pfor/ShapeShape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:
 loop_body/clip_by_value/pfor/subSub(loop_body/clip_by_value/pfor/Maximum:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: t
*loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¯
$loop_body/clip_by_value/pfor/ReshapeReshape$loop_body/clip_by_value/pfor/sub:z:03loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:­
!loop_body/clip_by_value/pfor/TileTile0loop_body/clip_by_value/pfor/Tile/input:output:0-loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
*loop_body/clip_by_value/pfor/strided_sliceStridedSlice+loop_body/clip_by_value/pfor/Shape:output:09loop_body/clip_by_value/pfor/strided_slice/stack:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
,loop_body/clip_by_value/pfor/strided_slice_1StridedSlice+loop_body/clip_by_value/pfor/Shape:output:0;loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
#loop_body/clip_by_value/pfor/concatConcatV23loop_body/clip_by_value/pfor/strided_slice:output:0*loop_body/clip_by_value/pfor/Tile:output:05loop_body/clip_by_value/pfor/strided_slice_1:output:01loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
&loop_body/clip_by_value/pfor/Reshape_1Reshape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0,loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
&loop_body/clip_by_value/pfor/Maximum_1Maximum/loop_body/clip_by_value/pfor/Reshape_1:output:0"loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity*loop_body/clip_by_value/pfor/Maximum_1:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp3^loop_body/stateful_uniform_full_int/RngReadAndSkip>^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2h
2loop_body/stateful_uniform_full_int/RngReadAndSkip2loop_body/stateful_uniform_full_int/RngReadAndSkip2~
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block3_conv1_layer_call_and_return_conditional_losses_14315

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
Ð
Ò
/loop_body_adjust_contrast_pfor_while_cond_15440Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1Z
Vloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_sliceq
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_15440___redundant_placeholder0q
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_15440___redundant_placeholder11
-loop_body_adjust_contrast_pfor_while_identity
Ü
)loop_body/adjust_contrast/pfor/while/LessLess0loop_body_adjust_contrast_pfor_while_placeholderVloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity-loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:


G__inference_block1_conv2_layer_call_and_return_conditional_losses_14262

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ö
¬
E__inference_sequential_layer_call_and_return_conditional_losses_16276
random_zoom_input
random_zoom_16163:	#
random_contrast_16166:	,
block1_conv1_16169:@ 
block1_conv1_16171:@,
block1_conv2_16174:@@ 
block1_conv2_16176:@-
block2_conv1_16180:@!
block2_conv1_16182:	.
block2_conv2_16185:!
block2_conv2_16187:	.
block3_conv1_16191:!
block3_conv1_16193:	.
block3_conv2_16196:!
block3_conv2_16198:	.
block3_conv3_16201:!
block3_conv3_16203:	.
block3_conv4_16206:!
block3_conv4_16208:	.
block4_conv1_16212:!
block4_conv1_16214:	.
block4_conv2_16217:!
block4_conv2_16219:	.
block4_conv3_16222:!
block4_conv3_16224:	.
block4_conv4_16227:!
block4_conv4_16229:	.
block5_conv1_16233:!
block5_conv1_16235:	.
block5_conv2_16238:!
block5_conv2_16240:	.
block5_conv3_16243:!
block5_conv3_16245:	.
block5_conv4_16248:!
block5_conv4_16250:	
dense_16255:
d
dense_16257:d'
batch_normalization_16260:d'
batch_normalization_16262:d'
batch_normalization_16264:d'
batch_normalization_16266:d
dense_1_16270:d
dense_1_16272:
identity¢+batch_normalization/StatefulPartitionedCall¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢'random_contrast/StatefulPartitionedCall¢#random_zoom/StatefulPartitionedCallù
#random_zoom/StatefulPartitionedCallStatefulPartitionedCallrandom_zoom_inputrandom_zoom_16163*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_15664 
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall,random_zoom/StatefulPartitionedCall:output:0random_contrast_16166*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_15549´
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0block1_conv1_16169block1_conv1_16171*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_14245±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_16174block1_conv2_16176*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_14262ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_14082¨
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_16180block2_conv1_16182*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_14280±
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_16185block2_conv2_16187*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_14297ð
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_14094§
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_16191block3_conv1_16193*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_14315°
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_16196block3_conv2_16198*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_14332°
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_16201block3_conv3_16203*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_14349°
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_16206block3_conv4_16208*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_14366ð
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_14106§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_16212block4_conv1_16214*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_14384°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_16217block4_conv2_16219*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_14401°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_16222block4_conv3_16224*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_14418°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_16227block4_conv4_16229*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_14435ð
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_14118§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_16233block5_conv1_16235*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_14453°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_16238block5_conv2_16240*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_14470°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_16243block5_conv3_16245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_14487°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_16248block5_conv4_16250*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_14504ð
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_14130Ø
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14517þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16255dense_16257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14529ô
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_16260batch_normalization_16262batch_normalization_16264batch_normalization_16266*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14204ì
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_14549
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_16270dense_1_16272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14562w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall(^random_contrast/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_zoom_input


ó
B__inference_dense_1_layer_call_and_return_conditional_losses_18995

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


G__inference_block5_conv4_layer_call_and_return_conditional_losses_18845

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs


G__inference_block4_conv4_layer_call_and_return_conditional_losses_14435

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
n
¿
F__inference_random_zoom_layer_call_and_return_conditional_losses_17772

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *333?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block3_conv2_layer_call_and_return_conditional_losses_14332

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
±
G
+__inference_block1_pool_layer_call_fn_18530

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_14082
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
¡
E__inference_sequential_layer_call_and_return_conditional_losses_15872

inputs
random_zoom_15759:	#
random_contrast_15762:	,
block1_conv1_15765:@ 
block1_conv1_15767:@,
block1_conv2_15770:@@ 
block1_conv2_15772:@-
block2_conv1_15776:@!
block2_conv1_15778:	.
block2_conv2_15781:!
block2_conv2_15783:	.
block3_conv1_15787:!
block3_conv1_15789:	.
block3_conv2_15792:!
block3_conv2_15794:	.
block3_conv3_15797:!
block3_conv3_15799:	.
block3_conv4_15802:!
block3_conv4_15804:	.
block4_conv1_15808:!
block4_conv1_15810:	.
block4_conv2_15813:!
block4_conv2_15815:	.
block4_conv3_15818:!
block4_conv3_15820:	.
block4_conv4_15823:!
block4_conv4_15825:	.
block5_conv1_15829:!
block5_conv1_15831:	.
block5_conv2_15834:!
block5_conv2_15836:	.
block5_conv3_15839:!
block5_conv3_15841:	.
block5_conv4_15844:!
block5_conv4_15846:	
dense_15851:
d
dense_15853:d'
batch_normalization_15856:d'
batch_normalization_15858:d'
batch_normalization_15860:d'
batch_normalization_15862:d
dense_1_15866:d
dense_1_15868:
identity¢+batch_normalization/StatefulPartitionedCall¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢'random_contrast/StatefulPartitionedCall¢#random_zoom/StatefulPartitionedCallî
#random_zoom/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_zoom_15759*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_15664 
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall,random_zoom/StatefulPartitionedCall:output:0random_contrast_15762*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_15549´
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0block1_conv1_15765block1_conv1_15767*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_14245±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_15770block1_conv2_15772*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_14262ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_14082¨
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_15776block2_conv1_15778*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_14280±
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_15781block2_conv2_15783*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_14297ð
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_14094§
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_15787block3_conv1_15789*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_14315°
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_15792block3_conv2_15794*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_14332°
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_15797block3_conv3_15799*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_14349°
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_15802block3_conv4_15804*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_14366ð
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_14106§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_15808block4_conv1_15810*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_14384°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_15813block4_conv2_15815*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_14401°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_15818block4_conv3_15820*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_14418°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_15823block4_conv4_15825*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_14435ð
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_14118§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_15829block5_conv1_15831*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_14453°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_15834block5_conv2_15836*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_14470°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_15839block5_conv3_15841*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_14487°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_15844block5_conv4_15846*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_14504ð
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_14130Ø
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14517þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_15851dense_15853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14529ô
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_15856batch_normalization_15858batch_normalization_15860batch_normalization_15862*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14204ì
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_14549
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_15866dense_1_15868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14562w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall(^random_contrast/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
G
+__inference_block4_pool_layer_call_fn_18760

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_14118
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
Ò
/loop_body_adjust_contrast_pfor_while_cond_18376Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1Z
Vloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_sliceq
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_18376___redundant_placeholder0q
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_18376___redundant_placeholder11
-loop_body_adjust_contrast_pfor_while_identity
Ü
)loop_body/adjust_contrast/pfor/while/LessLess0loop_body_adjust_contrast_pfor_while_placeholderVloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity-loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Å

*__inference_sequential_layer_call_fn_16543

inputs
unknown:	
	unknown_0:	#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	&

unknown_31:

unknown_32:	

unknown_33:
d

unknown_34:d

unknown_35:d

unknown_36:d

unknown_37:d

unknown_38:d

unknown_39:d

unknown_40:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$'()**0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_15872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapess
q:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë	
ó
@__inference_dense_layer_call_and_return_conditional_losses_14529

inputs2
matmul_readvariableop_resource:
d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
a
E__inference_activation_layer_call_and_return_conditional_losses_14549

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ú
¤
,__inference_block4_conv1_layer_call_fn_18684

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_14384x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÛF
Ì
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_16876­
¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter³
®random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterations]
Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_
[random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1ª
¥random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0ª
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0Z
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity\
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1\
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2\
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3¨
£random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¨
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1¢\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipÂ
\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkiprandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:
\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ü
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsdrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0erandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:â
rrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem[random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderarandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Srandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :´
Qrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Urandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Srandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ±
Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityWrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0S^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1Identity®random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsS^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ±
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2IdentityUrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0S^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ß
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identityrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0S^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ó
Rrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOp]^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "¹
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"½
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1arandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"½
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2arandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"½
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3arandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0"
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0"Î
£random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¥random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"º
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourcerandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2¼
\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


G__inference_block3_conv2_layer_call_and_return_conditional_losses_18625

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_14157

inputs*
cast_readvariableop_resource:d,
cast_1_readvariableop_resource:d,
cast_2_readvariableop_resource:d,
cast_3_readvariableop_resource:d
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:d*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:dm
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:dm
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¿g
Ë
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_15292
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0³
®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0·
²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0u
qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityU
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice±
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2µ
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1s
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderUloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdereloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
: 
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackNloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:­
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSlice®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0cloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
:  
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ï
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:¢
`loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0iloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¯
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSlice²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÑ
_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Î
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimshloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:Æ
kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderZloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ì
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Î
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: ¡
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identity loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ð
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: û
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identity{loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"ä
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shapeqloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0"
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_algloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"²
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"è
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0"à
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


G__inference_block1_conv1_layer_call_and_return_conditional_losses_18505

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_block4_conv1_layer_call_and_return_conditional_losses_14384

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿g
Ë
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_18228
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0³
®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0·
²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0u
qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityU
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice±
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2µ
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1s
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderUloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdereloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
: 
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackNloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:­
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSlice®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0cloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
:  
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ï
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:¢
`loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0iloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¯
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSlice²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÑ
_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Î
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimshloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:Æ
kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderZloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ì
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Î
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: ¡
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identity loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ð
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: û
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identity{loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"ä
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shapeqloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0"
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_algloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"²
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"è
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0"à
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
±
G
+__inference_block2_pool_layer_call_fn_18580

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_14094
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
·
?random_contrast_loop_body_adjust_contrast_pfor_while_cond_17382z
vrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counter
|random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterationsD
@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderF
Brandom_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1z
vrandom_contrast_loop_body_adjust_contrast_pfor_while_less_random_contrast_loop_body_adjust_contrast_pfor_strided_slice
random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_cond_17382___redundant_placeholder0
random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_cond_17382___redundant_placeholder1A
=random_contrast_loop_body_adjust_contrast_pfor_while_identity

9random_contrast/loop_body/adjust_contrast/pfor/while/LessLess@random_contrast_loop_body_adjust_contrast_pfor_while_placeholdervrandom_contrast_loop_body_adjust_contrast_pfor_while_less_random_contrast_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: ©
=random_contrast/loop_body/adjust_contrast/pfor/while/IdentityIdentity=random_contrast/loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
=random_contrast_loop_body_adjust_contrast_pfor_while_identityFrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:


G__inference_block3_conv4_layer_call_and_return_conditional_losses_18665

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
£
Î
3__inference_batch_normalization_layer_call_fn_18911

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


G__inference_block1_conv1_layer_call_and_return_conditional_losses_14245

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®>
å
/loop_body_adjust_contrast_pfor_while_body_15441Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1W
Sloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0Y
Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0h
dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_01
-loop_body_adjust_contrast_pfor_while_identity3
/loop_body_adjust_contrast_pfor_while_identity_13
/loop_body_adjust_contrast_pfor_while_identity_23
/loop_body_adjust_contrast_pfor_while_identity_3U
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceW
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2f
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2l
*loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(loop_body/adjust_contrast/pfor/while/addAddV20loop_body_adjust_contrast_pfor_while_placeholder3loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: |
:loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : å
8loop_body/adjust_contrast/pfor/while/strided_slice/stackPack0loop_body_adjust_contrast_pfor_while_placeholderCloop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:~
<loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : å
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_1Pack,loop_body/adjust_contrast/pfor/while/add:z:0Eloop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
2loop_body/adjust_contrast/pfor/while/strided_sliceStridedSliceUloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0Aloop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskn
,loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_1AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ~
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : é
:loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPack0loop_body_adjust_contrast_pfor_while_placeholderEloop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
>loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ë
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1Pack.loop_body/adjust_contrast/pfor/while/add_1:z:0Gloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
4loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicedloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0Cloop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_maskë
5loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2;loop_body/adjust_contrast/pfor/while/strided_slice:output:0=loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:u
3loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : î
/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDims>loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0<loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:¾
Iloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2loop_body_adjust_contrast_pfor_while_placeholder_10loop_body_adjust_contrast_pfor_while_placeholder8loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒn
,loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_2AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: n
,loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*loop_body/adjust_contrast/pfor/while/add_3AddV2Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter5loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity.loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: º
/loop_body/adjust_contrast/pfor/while/Identity_1Identity\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
/loop_body/adjust_contrast/pfor/while/Identity_2Identity.loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ·
/loop_body/adjust_contrast/pfor/while/Identity_3IdentityYloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_18loop_body/adjust_contrast/pfor/while/Identity_1:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_28loop_body/adjust_contrast/pfor/while/Identity_2:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_38loop_body/adjust_contrast/pfor/while/Identity_3:output:0"¨
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceSloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0"Ê
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0"¬
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ú
¤
,__inference_block3_conv4_layer_call_fn_18654

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_14366x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs
¯$
Ï
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14204

inputs5
'assignmovingavg_readvariableop_resource:d7
)assignmovingavg_1_readvariableop_resource:d*
cast_readvariableop_resource:d,
cast_1_readvariableop_resource:d
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:d*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:dx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:d¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:d*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:d~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:d´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:d*
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:d*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:dP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:dm
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:dc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:dk
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:dr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÞ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


E__inference_sequential_layer_call_and_return_conditional_losses_16160
random_zoom_input,
block1_conv1_16053:@ 
block1_conv1_16055:@,
block1_conv2_16058:@@ 
block1_conv2_16060:@-
block2_conv1_16064:@!
block2_conv1_16066:	.
block2_conv2_16069:!
block2_conv2_16071:	.
block3_conv1_16075:!
block3_conv1_16077:	.
block3_conv2_16080:!
block3_conv2_16082:	.
block3_conv3_16085:!
block3_conv3_16087:	.
block3_conv4_16090:!
block3_conv4_16092:	.
block4_conv1_16096:!
block4_conv1_16098:	.
block4_conv2_16101:!
block4_conv2_16103:	.
block4_conv3_16106:!
block4_conv3_16108:	.
block4_conv4_16111:!
block4_conv4_16113:	.
block5_conv1_16117:!
block5_conv1_16119:	.
block5_conv2_16122:!
block5_conv2_16124:	.
block5_conv3_16127:!
block5_conv3_16129:	.
block5_conv4_16132:!
block5_conv4_16134:	
dense_16139:
d
dense_16141:d'
batch_normalization_16144:d'
batch_normalization_16146:d'
batch_normalization_16148:d'
batch_normalization_16150:d
dense_1_16154:d
dense_1_16156:
identity¢+batch_normalization/StatefulPartitionedCall¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÕ
random_zoom/PartitionedCallPartitionedCallrandom_zoom_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_14226ð
random_contrast/PartitionedCallPartitionedCall$random_zoom/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_14232¬
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCall(random_contrast/PartitionedCall:output:0block1_conv1_16053block1_conv1_16055*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_14245±
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_16058block1_conv2_16060*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_14262ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿH@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_14082¨
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_16064block2_conv1_16066*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_14280±
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_16069block2_conv2_16071*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_14297ð
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_14094§
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_16075block3_conv1_16077*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_14315°
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_16080block3_conv2_16082*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_14332°
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_16085block3_conv3_16087*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_14349°
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_16090block3_conv4_16092*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_14366ð
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_14106§
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_16096block4_conv1_16098*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_14384°
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_16101block4_conv2_16103*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_14401°
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_16106block4_conv3_16108*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_14418°
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_16111block4_conv4_16113*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_14435ð
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_14118§
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_16117block5_conv1_16119*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_14453°
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_16122block5_conv2_16124*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_14470°
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_16127block5_conv3_16129*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_14487°
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_16132block5_conv4_16134*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_14504ð
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_14130Ø
flatten/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14517þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16139dense_16141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14529ö
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_16144batch_normalization_16146batch_normalization_16148batch_normalization_16150*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14157ì
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_14549
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0dense_1_16154dense_1_16156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_14562w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_zoom_input
²
C
'__inference_flatten_layer_call_fn_18860

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14517b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
a
E__inference_activation_layer_call_and_return_conditional_losses_18975

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Á
{
+__inference_random_zoom_layer_call_fn_17666

inputs
unknown:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_15664y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û]
Ä
erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_17165Ç
Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterÍ
Èrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsj
frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderl
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1l
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Ä
¿random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0«
¦random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0	g
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityi
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1i
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2i
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3i
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4Â
½random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice©
¤random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice	¢
`random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Û
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderirandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ²
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackfrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderyrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:´
rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1Packbrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0{random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:Á
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ï
hrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSlice¦random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask¬
wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterqrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::«
irandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDims}random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemhrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholdernrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ­
krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimsrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0trandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:
random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemhrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderprandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ¤
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ß
`random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderkrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ¤
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :¼
`random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterkrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ö
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitydrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: Ý
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1IdentityÈrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: ø
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2Identitydrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: ¤
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identityrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ¦
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identityrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "Ó
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identitylrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"
½random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¿random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"Ð
¤random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice¦random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

b
F__inference_block1_pool_layer_call_and_return_conditional_losses_14082

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
»
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_14998~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_14998___redundant_placeholder0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity
¤
;loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLessBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderzloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: ­
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
­
á	
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_17869
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_17869___redundant_placeholder0¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_17869___redundant_placeholder1¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_17869___redundant_placeholder2J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity
Á
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessIloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: »
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityFloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:


G__inference_block5_conv1_layer_call_and_return_conditional_losses_14453

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

b
F__inference_block5_pool_layer_call_and_return_conditional_losses_18855

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ý

erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_17164Ç
Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterÍ
Èrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsj
frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderl
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1l
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Ç
Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceÞ
Ùrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_17164___redundant_placeholder0	g
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
µ
_random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessfrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderÂrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: õ
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitycrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Ó
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identitylrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


G__inference_block3_conv3_layer_call_and_return_conditional_losses_18645

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ$@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ$@
 
_user_specified_nameinputs

b
F__inference_block3_pool_layer_call_and_return_conditional_losses_14106

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ØP

9loop_body_stateful_uniform_full_int_pfor_while_body_18059n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1k
gloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0^
Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0;
7loop_body_stateful_uniform_full_int_pfor_while_identity=
9loop_body_stateful_uniform_full_int_pfor_while_identity_1=
9loop_body_stateful_uniform_full_int_pfor_while_identity_2=
9loop_body_stateful_uniform_full_int_pfor_while_identity_3i
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZ
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algv
4loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :×
2loop_body/stateful_uniform_full_int/pfor/while/addAddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder=loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Bloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderMloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1Pack6loop_body/stateful_uniform_full_int/pfor/while/add:z:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
<loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0Kloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskx
6loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderOloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1Pack8loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0Qloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¥
>loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maské
Nloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0Eloop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0Gloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
=loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
9loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsWloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0Floop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
Sloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1:loop_body_stateful_uniform_full_int_pfor_while_placeholderBloop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐx
6loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: x
6loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
4loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_counter?loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity8loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: Ø
9loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identityploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
:  
9loop_body/stateful_uniform_full_int/pfor/while/Identity_2Identity8loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: Ë
9loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identitycloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_1Bloop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_2Bloop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_3Bloop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0"²
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algXloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0"Ð
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slicegloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"¶
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 


G__inference_block5_conv2_layer_call_and_return_conditional_losses_18805

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*È
serving_default´
Y
random_zoom_inputD
#serving_default_random_zoom_input:0ÿÿÿÿÿÿÿÿÿ;
dense_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:è
é
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer_with_weights-15
layer-22
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&
signatures"
_tf_keras_sequential
¼
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator"
_tf_keras_layer
¼
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator"
_tf_keras_layer
"
_tf_keras_input_layer
Ý
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
Ý
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op"
_tf_keras_layer
¥
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias
 U_jit_compiled_convolution_op"
_tf_keras_layer
Ý
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
¥
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op"
_tf_keras_layer
Ý
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias
 v_jit_compiled_convolution_op"
_tf_keras_layer
Ý
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias
 _jit_compiled_convolution_op"
_tf_keras_layer
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
! _jit_compiled_convolution_op"
_tf_keras_layer
æ
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§kernel
	¨bias
!©_jit_compiled_convolution_op"
_tf_keras_layer
æ
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses
°kernel
	±bias
!²_jit_compiled_convolution_op"
_tf_keras_layer
«
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"
_tf_keras_layer
æ
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses
¿kernel
	Àbias
!Á_jit_compiled_convolution_op"
_tf_keras_layer
æ
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses
Èkernel
	Ébias
!Ê_jit_compiled_convolution_op"
_tf_keras_layer
æ
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses
Ñkernel
	Òbias
!Ó_jit_compiled_convolution_op"
_tf_keras_layer
æ
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses
Úkernel
	Ûbias
!Ü_jit_compiled_convolution_op"
_tf_keras_layer
«
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses
ïkernel
	ðbias"
_tf_keras_layer
õ
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses
	÷axis

øgamma
	ùbeta
úmoving_mean
ûmoving_variance"
_tf_keras_layer
«
ü	variables
ýtrainable_variables
þregularization_losses
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
ð
;0
<1
D2
E3
S4
T5
\6
]7
k8
l9
t10
u11
}12
~13
14
15
16
17
18
19
§20
¨21
°22
±23
¿24
À25
È26
É27
Ñ28
Ò29
Ú30
Û31
ï32
ð33
ø34
ù35
ú36
û37
38
39"
trackable_list_wrapper
P
ï0
ð1
ø2
ù3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
æ
trace_0
trace_1
trace_2
trace_32ó
*__inference_sequential_layer_call_fn_14652
*__inference_sequential_layer_call_fn_16454
*__inference_sequential_layer_call_fn_16543
*__inference_sequential_layer_call_fn_16048À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ò
trace_0
trace_1
trace_2
trace_32ß
E__inference_sequential_layer_call_and_return_conditional_losses_16696
E__inference_sequential_layer_call_and_return_conditional_losses_17654
E__inference_sequential_layer_call_and_return_conditional_losses_16160
E__inference_sequential_layer_call_and_return_conditional_losses_16276À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
ÕBÒ
 __inference__wrapped_model_14073random_zoom_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü
	iter
beta_1
beta_2

decay
learning_rate	ïmö	ðm÷	ømø	ùmù	mú	mû	ïvü	ðvý	øvþ	ùvÿ	v	v"
	optimizer
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ì
¢trace_0
£trace_12
+__inference_random_zoom_layer_call_fn_17659
+__inference_random_zoom_layer_call_fn_17666´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¢trace_0z£trace_1

¤trace_0
¥trace_12Ç
F__inference_random_zoom_layer_call_and_return_conditional_losses_17670
F__inference_random_zoom_layer_call_and_return_conditional_losses_17772´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¤trace_0z¥trace_1
/
¦
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ô
¬trace_0
­trace_12
/__inference_random_contrast_layer_call_fn_17777
/__inference_random_contrast_layer_call_fn_17784´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¬trace_0z­trace_1

®trace_0
¯trace_12Ï
J__inference_random_contrast_layer_call_and_return_conditional_losses_17788
J__inference_random_contrast_layer_call_and_return_conditional_losses_18485´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z®trace_0z¯trace_1
/
°
_generator"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
ò
¶trace_02Ó
,__inference_block1_conv1_layer_call_fn_18494¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¶trace_0

·trace_02î
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18505¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
ò
½trace_02Ó
,__inference_block1_conv2_layer_call_fn_18514¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0

¾trace_02î
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18525¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
ñ
Ätrace_02Ò
+__inference_block1_pool_layer_call_fn_18530¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÄtrace_0

Åtrace_02í
F__inference_block1_pool_layer_call_and_return_conditional_losses_18535¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ò
Ëtrace_02Ó
,__inference_block2_conv1_layer_call_fn_18544¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zËtrace_0

Ìtrace_02î
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18555¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
ò
Òtrace_02Ó
,__inference_block2_conv2_layer_call_fn_18564¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÒtrace_0

Ótrace_02î
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18575¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÓtrace_0
/:-2block2_conv2/kernel
 :2block2_conv2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ñ
Ùtrace_02Ò
+__inference_block2_pool_layer_call_fn_18580¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÙtrace_0

Útrace_02í
F__inference_block2_pool_layer_call_and_return_conditional_losses_18585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÚtrace_0
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ò
àtrace_02Ó
,__inference_block3_conv1_layer_call_fn_18594¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zàtrace_0

átrace_02î
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18605¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zátrace_0
/:-2block3_conv1/kernel
 :2block3_conv1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ò
çtrace_02Ó
,__inference_block3_conv2_layer_call_fn_18614¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zçtrace_0

ètrace_02î
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18625¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zètrace_0
/:-2block3_conv2/kernel
 :2block3_conv2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
ò
îtrace_02Ó
,__inference_block3_conv3_layer_call_fn_18634¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zîtrace_0

ïtrace_02î
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18645¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zïtrace_0
/:-2block3_conv3/kernel
 :2block3_conv3/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ò
õtrace_02Ó
,__inference_block3_conv4_layer_call_fn_18654¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zõtrace_0

ötrace_02î
G__inference_block3_conv4_layer_call_and_return_conditional_losses_18665¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zötrace_0
/:-2block3_conv4/kernel
 :2block3_conv4/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
ütrace_02Ò
+__inference_block3_pool_layer_call_fn_18670¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zütrace_0

ýtrace_02í
F__inference_block3_pool_layer_call_and_return_conditional_losses_18675¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zýtrace_0
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Ó
,__inference_block4_conv1_layer_call_fn_18684¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02î
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18695¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
/:-2block4_conv1/kernel
 :2block4_conv1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Ó
,__inference_block4_conv2_layer_call_fn_18704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02î
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18715¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
/:-2block4_conv2/kernel
 :2block4_conv2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
§0
¨1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Ó
,__inference_block4_conv3_layer_call_fn_18724¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02î
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18735¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
/:-2block4_conv3/kernel
 :2block4_conv3/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Ó
,__inference_block4_conv4_layer_call_fn_18744¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02î
G__inference_block4_conv4_layer_call_and_return_conditional_losses_18755¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
/:-2block4_conv4/kernel
 :2block4_conv4/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_block4_pool_layer_call_fn_18760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

 trace_02í
F__inference_block4_pool_layer_call_and_return_conditional_losses_18765¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
ò
¦trace_02Ó
,__inference_block5_conv1_layer_call_fn_18774¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0

§trace_02î
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18785¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0
/:-2block5_conv1/kernel
 :2block5_conv1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
È0
É1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
ò
­trace_02Ó
,__inference_block5_conv2_layer_call_fn_18794¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z­trace_0

®trace_02î
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18805¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0
/:-2block5_conv2/kernel
 :2block5_conv2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
Ñ0
Ò1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
ò
´trace_02Ó
,__inference_block5_conv3_layer_call_fn_18814¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z´trace_0

µtrace_02î
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18825¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0
/:-2block5_conv3/kernel
 :2block5_conv3/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
0
Ú0
Û1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
ò
»trace_02Ó
,__inference_block5_conv4_layer_call_fn_18834¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z»trace_0

¼trace_02î
G__inference_block5_conv4_layer_call_and_return_conditional_losses_18845¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¼trace_0
/:-2block5_conv4/kernel
 :2block5_conv4/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
ñ
Âtrace_02Ò
+__inference_block5_pool_layer_call_fn_18850¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÂtrace_0

Ãtrace_02í
F__inference_block5_pool_layer_call_and_return_conditional_losses_18855¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÃtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
í
Étrace_02Î
'__inference_flatten_layer_call_fn_18860¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÉtrace_0

Êtrace_02é
B__inference_flatten_layer_call_and_return_conditional_losses_18866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÊtrace_0
0
ï0
ð1"
trackable_list_wrapper
0
ï0
ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
ë
Ðtrace_02Ì
%__inference_dense_layer_call_fn_18875¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÐtrace_0

Ñtrace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_18885¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÑtrace_0
 :
d2dense/kernel
:d2
dense/bias
@
ø0
ù1
ú2
û3"
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
Ü
×trace_0
Øtrace_12¡
3__inference_batch_normalization_layer_call_fn_18898
3__inference_batch_normalization_layer_call_fn_18911´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z×trace_0zØtrace_1

Ùtrace_0
Útrace_12×
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18931
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18965´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÙtrace_0zÚtrace_1
 "
trackable_list_wrapper
':%d2batch_normalization/gamma
&:$d2batch_normalization/beta
/:-d (2batch_normalization/moving_mean
3:1d (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
ü	variables
ýtrainable_variables
þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ð
àtrace_02Ñ
*__inference_activation_layer_call_fn_18970¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zàtrace_0

átrace_02ì
E__inference_activation_layer_call_and_return_conditional_losses_18975¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zátrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
çtrace_02Î
'__inference_dense_1_layer_call_fn_18984¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zçtrace_0

ètrace_02é
B__inference_dense_1_layer_call_and_return_conditional_losses_18995¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zètrace_0
 :d2dense_1/kernel
:2dense_1/bias
º
;0
<1
D2
E3
S4
T5
\6
]7
k8
l9
t10
u11
}12
~13
14
15
16
17
18
19
§20
¨21
°22
±23
¿24
À25
È26
É27
Ñ28
Ò29
Ú30
Û31
ú32
û33"
trackable_list_wrapper
þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
0
é0
ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_sequential_layer_call_fn_14652random_zoom_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
*__inference_sequential_layer_call_fn_16454inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
*__inference_sequential_layer_call_fn_16543inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
*__inference_sequential_layer_call_fn_16048random_zoom_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_16696inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_17654inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢B
E__inference_sequential_layer_call_and_return_conditional_losses_16160random_zoom_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢B
E__inference_sequential_layer_call_and_return_conditional_losses_16276random_zoom_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÔBÑ
#__inference_signature_wrapper_16369random_zoom_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
+__inference_random_zoom_layer_call_fn_17659inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñBî
+__inference_random_zoom_layer_call_fn_17666inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_random_zoom_layer_call_and_return_conditional_losses_17670inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_random_zoom_layer_call_and_return_conditional_losses_17772inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
/
ë
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
õBò
/__inference_random_contrast_layer_call_fn_17777inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õBò
/__inference_random_contrast_layer_call_fn_17784inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_random_contrast_layer_call_and_return_conditional_losses_17788inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_random_contrast_layer_call_and_return_conditional_losses_18485inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
/
ì
_state_var"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block1_conv1_layer_call_fn_18494inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18505inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block1_conv2_layer_call_fn_18514inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18525inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_block1_pool_layer_call_fn_18530inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_block1_pool_layer_call_and_return_conditional_losses_18535inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block2_conv1_layer_call_fn_18544inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18555inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block2_conv2_layer_call_fn_18564inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18575inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_block2_pool_layer_call_fn_18580inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_block2_pool_layer_call_and_return_conditional_losses_18585inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block3_conv1_layer_call_fn_18594inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18605inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block3_conv2_layer_call_fn_18614inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18625inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block3_conv3_layer_call_fn_18634inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18645inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block3_conv4_layer_call_fn_18654inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block3_conv4_layer_call_and_return_conditional_losses_18665inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_block3_pool_layer_call_fn_18670inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_block3_pool_layer_call_and_return_conditional_losses_18675inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block4_conv1_layer_call_fn_18684inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18695inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block4_conv2_layer_call_fn_18704inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18715inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
§0
¨1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block4_conv3_layer_call_fn_18724inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18735inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block4_conv4_layer_call_fn_18744inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block4_conv4_layer_call_and_return_conditional_losses_18755inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_block4_pool_layer_call_fn_18760inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_block4_pool_layer_call_and_return_conditional_losses_18765inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
¿0
À1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block5_conv1_layer_call_fn_18774inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18785inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
È0
É1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block5_conv2_layer_call_fn_18794inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18805inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
Ñ0
Ò1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block5_conv3_layer_call_fn_18814inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18825inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
Ú0
Û1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_block5_conv4_layer_call_fn_18834inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_block5_conv4_layer_call_and_return_conditional_losses_18845inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_block5_pool_layer_call_fn_18850inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_block5_pool_layer_call_and_return_conditional_losses_18855inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_flatten_layer_call_fn_18860inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_flatten_layer_call_and_return_conditional_losses_18866inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÙBÖ
%__inference_dense_layer_call_fn_18875inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
@__inference_dense_layer_call_and_return_conditional_losses_18885inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
ú0
û1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
3__inference_batch_normalization_layer_call_fn_18898inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ùBö
3__inference_batch_normalization_layer_call_fn_18911inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18931inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18965inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_activation_layer_call_fn_18970inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_activation_layer_call_and_return_conditional_losses_18975inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_dense_1_layer_call_fn_18984inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_1_layer_call_and_return_conditional_losses_18995inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
í	variables
î	keras_api

ïtotal

ðcount"
_tf_keras_metric
c
ñ	variables
ò	keras_api

ótotal

ôcount
õ
_fn_kwargs"
_tf_keras_metric
:	2StateVar
:	2StateVar
0
ï0
ð1"
trackable_list_wrapper
.
í	variables"
_generic_user_object
:  (2total
:  (2count
0
ó0
ô1"
trackable_list_wrapper
.
ñ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
%:#
d2Adam/dense/kernel/m
:d2Adam/dense/bias/m
,:*d2 Adam/batch_normalization/gamma/m
+:)d2Adam/batch_normalization/beta/m
%:#d2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#
d2Adam/dense/kernel/v
:d2Adam/dense/bias/v
,:*d2 Adam/batch_normalization/gamma/v
+:)d2Adam/batch_normalization/beta/v
%:#d2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/vâ
 __inference__wrapped_model_14073½B;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøD¢A
:¢7
52
random_zoom_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ¡
E__inference_activation_layer_call_and_return_conditional_losses_18975X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 y
*__inference_activation_layer_call_fn_18970K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿd¸
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18931fúûùø3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¸
N__inference_batch_normalization_layer_call_and_return_conditional_losses_18965fúûùø3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
3__inference_batch_normalization_layer_call_fn_18898Yúûùø3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿd
3__inference_batch_normalization_layer_call_fn_18911Yúûùø3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd»
G__inference_block1_conv1_layer_call_and_return_conditional_losses_18505p;<9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_block1_conv1_layer_call_fn_18494c;<9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ@»
G__inference_block1_conv2_layer_call_and_return_conditional_losses_18525pDE9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_block1_conv2_layer_call_fn_18514cDE9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@é
F__inference_block1_pool_layer_call_and_return_conditional_losses_18535R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block1_pool_layer_call_fn_18530R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
G__inference_block2_conv1_layer_call_and_return_conditional_losses_18555oST8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿH@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿH
 
,__inference_block2_conv1_layer_call_fn_18544bST8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿH@
ª ""ÿÿÿÿÿÿÿÿÿH»
G__inference_block2_conv2_layer_call_and_return_conditional_losses_18575p\]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿH
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿH
 
,__inference_block2_conv2_layer_call_fn_18564c\]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿH
ª ""ÿÿÿÿÿÿÿÿÿHé
F__inference_block2_pool_layer_call_and_return_conditional_losses_18585R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block2_pool_layer_call_fn_18580R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
G__inference_block3_conv1_layer_call_and_return_conditional_losses_18605nkl8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ$@
 
,__inference_block3_conv1_layer_call_fn_18594akl8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª "!ÿÿÿÿÿÿÿÿÿ$@¹
G__inference_block3_conv2_layer_call_and_return_conditional_losses_18625ntu8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ$@
 
,__inference_block3_conv2_layer_call_fn_18614atu8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª "!ÿÿÿÿÿÿÿÿÿ$@¹
G__inference_block3_conv3_layer_call_and_return_conditional_losses_18645n}~8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ$@
 
,__inference_block3_conv3_layer_call_fn_18634a}~8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª "!ÿÿÿÿÿÿÿÿÿ$@»
G__inference_block3_conv4_layer_call_and_return_conditional_losses_18665p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ$@
 
,__inference_block3_conv4_layer_call_fn_18654c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ$@
ª "!ÿÿÿÿÿÿÿÿÿ$@é
F__inference_block3_pool_layer_call_and_return_conditional_losses_18675R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block3_pool_layer_call_fn_18670R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
G__inference_block4_conv1_layer_call_and_return_conditional_losses_18695p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_block4_conv1_layer_call_fn_18684c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª "!ÿÿÿÿÿÿÿÿÿ »
G__inference_block4_conv2_layer_call_and_return_conditional_losses_18715p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_block4_conv2_layer_call_fn_18704c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª "!ÿÿÿÿÿÿÿÿÿ »
G__inference_block4_conv3_layer_call_and_return_conditional_losses_18735p§¨8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_block4_conv3_layer_call_fn_18724c§¨8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª "!ÿÿÿÿÿÿÿÿÿ »
G__inference_block4_conv4_layer_call_and_return_conditional_losses_18755p°±8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_block4_conv4_layer_call_fn_18744c°±8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ 
ª "!ÿÿÿÿÿÿÿÿÿ é
F__inference_block4_pool_layer_call_and_return_conditional_losses_18765R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block4_pool_layer_call_fn_18760R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
G__inference_block5_conv1_layer_call_and_return_conditional_losses_18785p¿À8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ	
 
,__inference_block5_conv1_layer_call_fn_18774c¿À8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª "!ÿÿÿÿÿÿÿÿÿ	»
G__inference_block5_conv2_layer_call_and_return_conditional_losses_18805pÈÉ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ	
 
,__inference_block5_conv2_layer_call_fn_18794cÈÉ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª "!ÿÿÿÿÿÿÿÿÿ	»
G__inference_block5_conv3_layer_call_and_return_conditional_losses_18825pÑÒ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ	
 
,__inference_block5_conv3_layer_call_fn_18814cÑÒ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª "!ÿÿÿÿÿÿÿÿÿ	»
G__inference_block5_conv4_layer_call_and_return_conditional_losses_18845pÚÛ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ	
 
,__inference_block5_conv4_layer_call_fn_18834cÚÛ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ	
ª "!ÿÿÿÿÿÿÿÿÿ	é
F__inference_block5_pool_layer_call_and_return_conditional_losses_18855R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Á
+__inference_block5_pool_layer_call_fn_18850R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_1_layer_call_and_return_conditional_losses_18995^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_1_layer_call_fn_18984Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ¤
@__inference_dense_layer_call_and_return_conditional_losses_18885`ïð1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 |
%__inference_dense_layer_call_fn_18875Sïð1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿd©
B__inference_flatten_layer_call_and_return_conditional_losses_18866c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_flatten_layer_call_fn_18860V8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¾
J__inference_random_contrast_layer_call_and_return_conditional_losses_17788p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_random_contrast_layer_call_and_return_conditional_losses_18485tì=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
/__inference_random_contrast_layer_call_fn_17777c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
/__inference_random_contrast_layer_call_fn_17784gì=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿº
F__inference_random_zoom_layer_call_and_return_conditional_losses_17670p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¾
F__inference_random_zoom_layer_call_and_return_conditional_losses_17772të=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_random_zoom_layer_call_fn_17659c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
+__inference_random_zoom_layer_call_fn_17666gë=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ
E__inference_sequential_layer_call_and_return_conditional_losses_16160¹B;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøL¢I
B¢?
52
random_zoom_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
E__inference_sequential_layer_call_and_return_conditional_losses_16276½Fëì;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøL¢I
B¢?
52
random_zoom_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ø
E__inference_sequential_layer_call_and_return_conditional_losses_16696®B;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
E__inference_sequential_layer_call_and_return_conditional_losses_17654²Fëì;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Û
*__inference_sequential_layer_call_fn_14652¬B;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøL¢I
B¢?
52
random_zoom_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿß
*__inference_sequential_layer_call_fn_16048°Fëì;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøL¢I
B¢?
52
random_zoom_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÐ
*__inference_sequential_layer_call_fn_16454¡B;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÔ
*__inference_sequential_layer_call_fn_16543¥Fëì;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿú
#__inference_signature_wrapper_16369ÒB;<DEST\]kltu}~§¨°±¿ÀÈÉÑÒÚÛïðúûùøY¢V
¢ 
OªL
J
random_zoom_input52
random_zoom_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ