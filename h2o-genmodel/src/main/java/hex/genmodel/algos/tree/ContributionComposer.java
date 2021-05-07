package hex.genmodel.algos.tree;

import hex.genmodel.utils.ArrayUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;

public class ContributionComposer implements Serializable {
    
    /**
     * Sort shapley values and compose desired output
     *
     * @param contribs Raw contributions to be composed
     * @param contribNameIds Contribution corresponding feature ids
     * @param topN Return only #topN highest contributions + bias.
     * @param topBottomN Return only #topBottomN lowest contributions + bias
     *                   If topN and topBottomN are defined together then return array of #topN + #topBottomN + bias
     * @param abs True to compare absolute values of contributions
     * @return Sorted KeyValue array of contributions of size #topN + #topBottomN + bias
     *         If topN < 0 || topBottomN < 0 then all descending sorted contributions is returned.
     */
    public final Integer[] composeContributions(final float[] contribs, Integer[] contribNameIds, int topN, int topBottomN, boolean abs) {
        if (topBottomN == 0) {
            return composeSortedContributions(contribs, contribNameIds, topN, abs, -1);
        } else if (topN == 0) {
            return composeSortedContributions(contribs, contribNameIds, topBottomN, abs,1);
        } else if ((topN + topBottomN) >= contribs.length || topN < 0 || topBottomN < 0) {
            return composeSortedContributions(contribs, contribNameIds, contribs.length, abs, -1);
        }

        composeSortedContributions(contribs, contribNameIds, contribNameIds.length, abs,-1);
        Integer[] bottomSorted = Arrays.copyOfRange(contribNameIds, contribNameIds.length - 1 - topBottomN, contribNameIds.length);
        reverse(bottomSorted, contribs, bottomSorted.length - 1);
        contribNameIds = Arrays.copyOf(contribNameIds, topN);

        contribNameIds = ArrayUtils.appendGeneric(contribNameIds, bottomSorted);
        return contribNameIds;
    }
    
    public int checkAndAdjustInput(int n, int len) {
        if (n < 0 || n > len) {
            return len;
        }
        return n;
    }
    
    private Integer[] composeSortedContributions(float[] contribs, Integer[] contribNameIds, int n, boolean abs, int increasing) {
        int nAdjusted = checkAndAdjustInput(n, contribs.length);
        sortContributions(contribs, contribNameIds, abs, increasing);
        if (nAdjusted < contribs.length) {
            Integer bias = contribNameIds[contribs.length-1];
            contribNameIds = Arrays.copyOfRange(contribNameIds, 0, nAdjusted + 1);
            contribNameIds[nAdjusted] = bias;
        }
        return contribNameIds;
    }
    
    private void sortContributions(final float[] contribs, Integer[] contribNameIds, final boolean abs, final int increasing) {
        Arrays.sort(contribNameIds, 0, contribs.length -1 /*exclude bias*/, new Comparator<Integer>() {
            @Override
            public int compare(Integer x, Integer y) {
                if (abs) {
                    return Float.compare(Math.abs(contribs[x]) * increasing, Math.abs(contribs[y]) * increasing);
                }
                else {
                    return Float.compare(contribs[x] * increasing, contribs[y] * increasing);
                }
            }
        });
    }

    private void reverse(Integer[] contribNameIds, float[] contribs, int len) {
        for (int i = 0; i < len/2; i++) {
            if (contribs[contribNameIds[i]] != contribs[contribNameIds[len - i - 1]]) {
                Integer tmp = contribNameIds[i];
                contribNameIds[i] = contribNameIds[len - i - 1];
                contribNameIds[len - i - 1] = tmp;
            }
        }
    }
}
