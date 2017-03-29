
package ia05regressao;

import java.io.File;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


/**
 *
 * @author Paulo Valle
 */
public class IA05Regressao {

    public static void main(String[] args) throws Exception {
        
        // inicia o leitor de csv
        CSVLoader leitor = new CSVLoader();
        
        // seta o caracter separador do csv
        leitor.setFieldSeparator(",");
        
        // seta o arquivo csv com os dados
        leitor.setSource(new File("data/ENB2012_data.csv"));
        
        // carrega os dados
        Instances dados = leitor.getDataSet();
        
        // System.out.println(dados);
        
        // seta o index para a penultima coluna
        dados.setClassIndex(dados.numAttributes() - 2);
        
        // aplica a remocao do item selecionado
        Remove remover = new Remove();
        remover.setOptions(new String[] {"-R", dados.numAttributes() + ""});
        remover.setInputFormat(dados);
        dados = Filter.useFilter(dados, remover);
        
        // aplica o algoritmo de regressão linear
        System.out.println("\n\nRESULTADOS DA REGRESSÃO LINEAR");
        LinearRegression modelo = new LinearRegression();
        modelo.buildClassifier(dados);
        System.out.println(modelo);
        
        // mostra as taxas de eficiencia do algoritmo
        System.out.println("\n\nTAXA DE RESULTADOS DO ALGORITMO");
        Evaluation validacao = new Evaluation(dados);
	validacao.crossValidateModel(modelo, dados, 10, 
			new Random(1), new String[] {});
	System.out.println(validacao.toSummaryString());

        
    }
    
}
