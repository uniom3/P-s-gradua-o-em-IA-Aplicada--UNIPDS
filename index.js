import * as tf from '@tensorflow/tfjs';


//80 neurônios porque tem pouca base de treino
//quanto mais neurônios, mais complexidade a rede pode aprender, mas também aumenta o risco de overfitting (quando o modelo se ajusta demais aos dados de treino e tem dificuldade em generalizar para novos dados).
//e consequentemente, mais processamento ela vai usar.
//A Relu age como um filtro:
//É como se ela deixasse somente os dados interessantes seguirem viagem na rede
//Se a informação chegou nesse neuronio é positiva, passa para frente!
//Se for zerou ou negativa, não tem nada de interessante ali, então a gente descarta aquela informação, e ela não segue viagem na rede.
async function trainModel(inputXs, outputYs) {
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

//Saída tem 3 neurônios porque tem 3 categorias (premium, medium, basic)
//activation 'softmax' é usada para problemas de classificação multiclasse, onde queremos que a saída seja uma distribuição de probabilidade sobre as classes. Ela transforma os valores de saída em probabilidades, garantindo que a soma das saídas seja igual a 1.
model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

//Compilando o modelo, optimizer Adam (adaptive Moment Estimation) é um trinador pessoal moderno para redes neurais:
//ajusta os pesos de forma eficiente e inteligente aprender com histórico de erros e acertos
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']   
});

//verbose 0 para não mostrar o progresso do treinamento, epochs 100 para treinar por 100 ciclos, shuffle true para embaralhar os dados a cada época, callbacks para mostrar a perda (loss) a cada época.
//epochs é o número de vezes que o modelo vai passar por todo o conjunto de dados de treinamento. Quanto mais épocas, mais o modelo pode aprender, mas também aumenta o risco de overfitting. É importante monitorar a perda e a precisão durante o treinamento para evitar esse problema.
//shuffle é uma técnica que embaralha os dados de treinamento a cada época. Isso ajuda a evitar que o modelo aprenda padrões específicos da ordem dos dados, o que pode levar a um melhor desempenho e generalização.

await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch}: loss = ${logs.loss}`); //, accuracy = ${logs.acc.toFixed(4)}
        }
    }

});
return model;
}

async function predict(model, pessoaTensorNormalizada) {
   //transformar o array em tensor
   const tfInput = tf.tensor2d(pessoaTensorNormalizada);
   const prediction = model.predict(tfInput);
   const predictionArray = await prediction.array();
   return predictionArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]


// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.

const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0,    0, 1, 0, 0, 1, 0],    // Ana
    [1,    0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];


// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs)

const pessoa = {nome: "zé", idade: 28, cor: "verde", localizacao: "São Paulo"}
//normalizando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: Se a idade minima era 20 e a máxima 40, então (28 -25)/(40-25) = 3/15 = 0.2
const pessoaTensorNormalizada = [[0.22, 1, 0, 0, 0, 1, 0]];

const probabilidade = await predict(model, pessoaTensorNormalizada);
const results = probabilidade.sort((a, b) => b.prob - a.prob)
.map(p =>`${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
.join("\n ");
console.log(results);


//inputXs.print();
//outputYs.print();