/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekafrontend;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Marcos Nava
 */
public class Classificacao {
    private DataSource arquivo;
    private Instances dados;
    private Remove filtro;
    private AttributeSelection selAtributo;
    private InfoGainAttributeEval avaliador;
    private Ranker busca;
    private J48 arvore;

    public Classificacao() {
        try
        {
            arquivo = new DataSource("dados/zoo.arff");
            dados = arquivo.getDataSet();
            
            String[] parametros = new String[] { "-R", "1" };
            filtro = new Remove();
            filtro.setOptions(parametros);
            filtro.setInputFormat(dados);
            dados = Filter.useFilter(dados, filtro);
            
            selAtributo = new AttributeSelection();
            avaliador = new InfoGainAttributeEval();
            busca = new Ranker();
            selAtributo.setEvaluator(avaliador);
            selAtributo.setSearch(busca);
            selAtributo.SelectAttributes(dados);
            
            String[] opcoes = new String[1];
            opcoes[0] = "-U";
            arvore = new J48();
            arvore.setOptions(opcoes);
            arvore.buildClassifier(dados);
        }
        catch(Exception e)
        {
            System.err.println(e);
        }
        
    }
    
    String classificar(double[] vals)
    {
        String retorno = "";
        try
        {
            Instance novoAnimal = new DenseInstance(1.0, vals);

            novoAnimal.setDataset(dados); 

            double label = arvore.classifyInstance(novoAnimal);
            retorno = dados.classAttribute().value((int) label);
        }
        catch(Exception e)
        {
            System.err.println(e);
        }
        return retorno;
    }
}
