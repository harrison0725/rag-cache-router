# Business Admin Console

The admin console is available on the Business plan at admin.lumendrive.example. Only users with the Admin or Billing Admin role can open it.

Adding users: admins can invite users by email, add them in bulk with a CSV file, or provision them automatically with SCIM from Okta, Microsoft Entra ID, or Google Workspace.

Removing users: when an admin removes a user, the admin chooses to transfer that user's files to another team member or to keep them in a suspended account for 30 days. After 30 days the suspended account and its files are deleted.

Roles: Admin (full control), Billing Admin (billing and invoices only), and Member. A team must always have at least one Admin.

Single sign-on: SSO supports any SAML 2.0 identity provider. When SSO is set to "Required", users can no longer sign in with a password, except Admins, who keep a password as a break-glass login.

Audit logs record sign-ins, sharing changes, file deletions, and admin actions. Logs are kept for 1 year and can be exported as CSV or streamed to a SIEM through the Events API.

Storage pooling: on Business, storage is pooled across the team. A team of 10 users has 50 TB shared by everyone, so one person can use more than 5 TB as long as the team total stays under the pool.
