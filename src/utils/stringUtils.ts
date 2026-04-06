/**
 * Return address text up to and including the first comma.
 * If the input contains no comma, return the original text.
 */
export function truncateToFirstComma(value: string): string {
  const firstCommaIndex = value.indexOf(',');
  if (firstCommaIndex === -1) {
    return value;
  }

  return value.slice(0, firstCommaIndex + 1).trimEnd();
}

/**
 * Return address text up to (but not including) the second comma.
 * If the input contains fewer than two commas, return the original text.
 */
export function truncateToSecondComma(value: string): string {
  const firstCommaIndex = value.indexOf(',');
  if (firstCommaIndex === -1) {
    return value;
  }

  const secondCommaIndex = value.indexOf(',', firstCommaIndex + 1);
  if (secondCommaIndex === -1) {
    return value;
  }

  return value.slice(0, secondCommaIndex).trimEnd();
}
