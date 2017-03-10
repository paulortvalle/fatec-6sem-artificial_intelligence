/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ia02classificacao;

import javax.swing.JFrame;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.J48;
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
        TreeVisualizer tv = new TreeVisualizer(null, arvore.graph(), new PlaceNode2());
        JFrame frame = new javax.swing.JFrame("Árvore de Conhecimento");
        frame.setSize(800,500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(tv);
        frame.setVisible(true);
        tv.fitToScreen();
    }
    
}
