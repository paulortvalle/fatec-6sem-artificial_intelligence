/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ia02classificacao;

import java.util.Random;
import javax.swing.JFrame;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;

import weka.gui.treevisualizer.TreeVisualizer;

/**
 *
 * @author Paulo Valle
 */
public class IA02Classificacao {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        // abre o banco de dados arff e mostra a quantidade de instancias (linhas)
        DataSource arquivo = new DataSource("data/zoo.arff");
        Instances dados = arquivo.getDataSet();
        System.out.println("Instancias lidas: " + dados.numInstances());    
        
        // FILTER: remove o atributo nome do animal da classificação
        String[] parametros = new String[]{"-R","1"};
        Remove filtro = new Remove();
        filtro.setOptions(parametros);
        filtro.setInputFormat(dados);
        dados = Filter.useFilter(dados, filtro);
                
        AttributeSelection selAtributo = new AttributeSelection();
        InfoGainAttributeEval avaliador = new InfoGainAttributeEval();
        Ranker busca = new Ranker();
        selAtributo.setEvaluator(avaliador);
        selAtributo.setSearch(busca);
        selAtributo.SelectAttributes(dados);
        int[] indices = selAtributo.selectedAttributes();
        System.out.println("Selected attributes: " + Utils.arrayToString(indices));
        
        // Usa o algoritimo J48 e mostra a classificação dos dados em forma textual
        String[] opcoes = new String[1];
        opcoes[0] = "-U";
        J48 arvore = new J48();
        arvore.setOptions(opcoes);
        arvore.buildClassifier(dados);
        System.out.println(arvore);
        
        // Usa o algoritimo J48 e mostra a classificação de dados em forma grafica
        /*
        TreeVisualizer tv = new TreeVisualizer(null, arvore.graph(), new PlaceNode2());
        JFrame frame = new javax.swing.JFrame("Árvore de Conhecimento");
        frame.setSize(800,500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(tv);
        frame.setVisible(true);
        tv.fitToScreen();
        */
        
        /*
        * Classificação de novos dados
        */
        
        System.out.println("\n\nCLASSIFICAÇÃO DE NOVOS DADOS");
        // criar atributos
        double[] vals = new double[dados.numAttributes()];
        vals[0] = 1.0; // hair
        vals[1] = 0.0; // feathers
        vals[2] = 0.0; // eggs
        vals[3] = 1.0; // milk
        vals[4] = 1.0; // airborne
        vals[5] = 0.0; // aquatic
        vals[6] = 0.0; // predator
        vals[7] = 1.0; // toothed
        vals[8] = 1.0; // backbone
        vals[9] = 1.0; // breathes
        vals[10] = 0.0; // venomous
        vals[11] = 0.0; // fins
        vals[12] = 4.0; // legs
        vals[13] = 1.0; // tail
        vals[14] = 1.0; // domestic
        vals[15] = 1.0; // catsize
        
        // Criar uma instância baseada nestes atributos
        Instance meuUnicornio = new DenseInstance(1.0, vals);

        // Adicionar a instância nos dados
        meuUnicornio.setDataset(dados);

        // Classificar esta nova instância
        double label = arvore.classifyInstance(meuUnicornio);

        // Imprimir o resultado da classificação
        System.out.println("Novo Animal: Unicórnio");
        System.out.println("classificacao: " + dados.classAttribute().value((int) label));
        
        
        
        /*
        * Avaliação e predição de erros de métrica
        */
        System.out.println("\n\nAVALIAÇÃO E PREDIÇÃO DE ERROS DE MÉTRICA");
        Classifier cl = new J48();
        Evaluation eval_roc = new Evaluation(dados);
        eval_roc.crossValidateModel(cl, dados, 10, new Random(1), new Object[] {});
        System.out.println(eval_roc.toSummaryString());
        
        /*
        * Matriz de confusão
        */
        System.out.println("\n\nMATRIZ DE CONFUSÃO");
        double[][] confusionMatrix = eval_roc.confusionMatrix();
        System.out.println(eval_roc.toMatrixString());
        
    } // main
    
} // class
