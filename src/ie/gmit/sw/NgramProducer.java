package ie.gmit.sw;

import java.util.Iterator;

/**
 * Custom ngram iterator to split strings into size n.
 * Used to iterate over Language Library and user text
 *
 * @author David Gallagher
 */
public class NgramProducer implements Iterator<String> {

    private final String str;
    private final int n;
    private int pos = 0;
    private int count = 0;

    /**
     * Creates a new <code>NgramIterator</code> object based on the
     * specified ngram size, and string.
     * @param n The size of the ngram
     * @param str The string to iterate.
     * */
    public NgramProducer(int n, String str) {
        this.n = n;
        this.str = str;
    }

    public int getCount(){
        return count;
    }
    /**
     * Returns true if, considering the ngram size, the iteration has more elements.
     * @return boolean true if the iteration has more elements.
     */
    public boolean hasNext() {
        return pos < str.length() - n;
    }

    /**
     * Returns the next element in the iteration.
     * @return String The next ngram string in the iterator.
     */
    public String next() {
        count++;
        return str.substring(pos, pos++ + n);
    }

    /**
     * Returns the hashcode value of the element in the iteration.
     * @return int The hashcode value of the ngram element in the iterator.
     */
    @Override
    public int hashCode() {
        count++;
        return str.substring(pos, pos++ + n).hashCode();
    }
}
