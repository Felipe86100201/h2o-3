package hex.genmodel.attributes.parameters;

import java.util.Comparator;

public class KeyValue {
    
    public final String key;
    public final double value;

    public KeyValue(String key, double value) {
        this.key = key;
        this.value = value;
    }

    public String getKey() {
        return key;
    }

    public double getValue() {
        return value;
    }

    @Override
    public String toString() {
        return "{Key: " + key + ", Value: " + value + "}";
    }
    
    public static class AscComparator implements Comparator<KeyValue> {
        private final boolean abs;
    
        public AscComparator(boolean abs) {
            this.abs = abs;
        }
        
        @Override
        public int compare(KeyValue o1, KeyValue o2) {
            if (abs)
                return Math.abs(o1.getValue()) < Math.abs(o2.getValue()) ? -1 : 0;
            return o1.getValue() < o2.getValue() ? -1 : 0;
        }
    }
    
    public static class DescComparator implements Comparator<KeyValue> {
        private final boolean abs;
        
        public DescComparator(boolean abs) {
            this.abs = abs;
        }
    
        @Override
        public int compare(KeyValue o1, KeyValue o2) {
            if (abs)
                return Math.abs(o1.getValue()) > Math.abs(o2.getValue()) ? -1 : 0;
            return o1.getValue() > o2.getValue() ? -1 : 0;
        }
    }
}
