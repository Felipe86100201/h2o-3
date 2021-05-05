package hex.security;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.security.UserGroupInformation;
import water.AbstractH2OExtension;
import water.H2O;
import water.persist.PersistHdfs;
import water.persist.security.HdfsDelegationTokenRefresher;
import water.util.Log;

import java.io.IOException;

/**
 * Authenticates the H2O Node to access secured Hadoop cluster in a standalone mode.
 *
 * This extension assumes that if Hadoop configuration is present and it has Kerberos enabled
 * the user will likely want to read data from HDFS even though H2O is running in a standalone mode (not on Hadoop).
 * The extension attempts to authenticate the user using an existing Kerberos ticket. This means the Kerberos ticket
 * needs to be manually acquired by the user on each node before the H2O instance is started.
 *
 * The extension fails gracefully if the user cannot be authenticated and doesn't stop H2O start-up. The failure
 * will be logged as an error.
 */
public class KerberosExtension extends AbstractH2OExtension {

  public static String NAME = "KrbStandalone";

  @Override
  public String getExtensionName() {
    return NAME;
  }

  @Override
  public void onLocalNodeStarted() {
    Configuration conf = PersistHdfs.CONF;
    if (conf == null)
      return; // this is theoretically possible although unlikely

    if (isKerberosEnabled(conf)) {
      UserGroupInformation.setConfiguration(conf);
      final UserGroupInformation ugi;
      if (H2O.ARGS.keytab_path != null) {
        Log.debug("Kerberos enabled in Hadoop configuration. Trying to login user from keytab.");
        ugi = loginUserFromKeytab(H2O.ARGS.principal, H2O.ARGS.keytab_path);
      } else {
        Log.debug("Kerberos enabled in Hadoop configuration. Trying to login the (default) user.");
        ugi = loginDefaultUser();
      }
      if (ugi != null) {
        Log.info("Kerberos subsystem initialized. Using user '" + ugi.getShortUserName() + "'.");
      }
      if (H2O.ARGS.hdfs_token_refresh) {
        HdfsDelegationTokenRefresher.startRefresher(conf, H2O.ARGS.principal, H2O.ARGS.keytab_path, null);
      }
    } else
      Log.debug("Kerberos not configured");
  }

  private UserGroupInformation loginDefaultUser() {
    try {
      UserGroupInformation.loginUserFromSubject(null);
      return UserGroupInformation.getCurrentUser();
    } catch (IOException e) {
      Log.err("Kerberos initialization FAILED. Kerberos ticket needs to be acquired before starting H2O (run kinit).", e);
      return null;
    }
  }

  private static UserGroupInformation loginUserFromKeytab(String authPrincipal, String authKeytabPath) {
    try {
      return UserGroupInformation.loginUserFromKeytabAndReturnUGI(authPrincipal, authKeytabPath);
    } catch (IOException e) {
      throw new RuntimeException("Failed to login user " + authPrincipal + " from keytab " + authKeytabPath);
    }
  }

  private static boolean isKerberosEnabled(Configuration conf) {
    return "kerberos".equals(conf.get("hadoop.security.authentication"));
  }

}
